use chrono::Utc;

use crate::Rating;
use crate::alea;

/// Minimum stability value (days). FSRS-6 clamps stability to at least this.
pub const STABILITY_MIN: f64 = 0.001;
/// Maximum stability value (days). 100 years.
pub const STABILITY_MAX: f64 = 36500.0;
/// Minimum difficulty value.
pub const DIFFICULTY_MIN: f64 = 1.0;
/// Maximum difficulty value.
pub const DIFFICULTY_MAX: f64 = 10.0;

/// Default decay parameter for the FSRS-6 power forgetting curve.
/// Lives at `w[20]` in the parameter vector.
pub const FSRS6_DEFAULT_DECAY: f64 = 0.1542;

type Weights = [f64; 21];

/// FSRS-6 default weights, taken from open-spaced-repetition/fsrs-rs
/// (`src/inference.rs::DEFAULT_PARAMETERS`).
const DEFAULT_WEIGHTS: Weights = [
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    FSRS6_DEFAULT_DECAY,
];

#[derive(Debug, Clone)]
pub struct Parameters {
    pub request_retention: f64,
    pub maximum_interval: i32,
    pub w: Weights,
    pub enable_short_term: bool,
    pub enable_fuzz: bool,
    pub seed: Seed,
}

impl Parameters {
    /// Decay parameter for the forgetting curve, derived from `w[20]`.
    /// (FSRS-6 makes the decay a learnable parameter; FSRS-5 hardcoded it.)
    #[inline]
    pub fn decay(&self) -> f64 {
        -self.w[20]
    }

    /// Factor used in the forgetting curve, derived from decay.
    /// `decay` is negative (e.g. -0.1542 with FSRS-6 defaults), so this expands
    /// to `0.9.powf(negative)` which is > 1 — yielding a positive factor.
    #[inline]
    pub fn factor(&self) -> f64 {
        let decay = self.decay();
        f64::powf(0.9, 1.0 / decay) - 1.0
    }

    /// Probability of recalling the card at `elapsed_days` after the last review,
    /// given the card's `stability`.
    pub fn forgetting_curve(&self, elapsed_days: f64, stability: f64) -> f64 {
        let decay = self.decay();
        let factor = self.factor();
        (1.0 + factor * elapsed_days / stability).powf(decay)
    }

    pub fn init_difficulty(&self, rating: Rating) -> f64 {
        let rating_int: i32 = rating as i32;
        (self.w[4] - f64::exp(self.w[5] * (rating_int as f64 - 1.0)) + 1.0)
            .clamp(DIFFICULTY_MIN, DIFFICULTY_MAX)
    }

    pub fn init_stability(&self, rating: Rating) -> f64 {
        let rating_int: i32 = rating as i32;
        self.w[(rating_int - 1) as usize].max(STABILITY_MIN)
    }

    #[allow(clippy::suboptimal_flops)]
    pub fn next_interval(&self, stability: f64, elapsed_days: i64) -> f64 {
        let decay = self.decay();
        let factor = self.factor();
        let new_interval = (stability / factor * (self.request_retention.powf(1.0 / decay) - 1.0))
            .round()
            .clamp(1.0, self.maximum_interval as f64);
        self.apply_fuzz(new_interval, elapsed_days)
    }

    pub fn next_difficulty(&self, difficulty: f64, rating: Rating) -> f64 {
        let rating_int = rating as i32;
        let delta_d = -self.w[6] * (rating_int as f64 - 3.0);
        // FSRS-6 linear damping: difficulty changes less when the card is already
        // near the extremes (very easy or very hard).
        let damped = difficulty + Self::linear_damping(delta_d, difficulty);
        let mean_reverted = self.mean_reversion(self.init_difficulty(Rating::Easy), damped);
        mean_reverted.clamp(DIFFICULTY_MIN, DIFFICULTY_MAX)
    }

    fn linear_damping(delta_d: f64, old_d: f64) -> f64 {
        (10.0 - old_d) * delta_d / 9.0
    }

    pub fn short_term_stability(&self, stability: f64, rating: Rating) -> f64 {
        let rating_int = rating as i32;
        let mut sinc = f64::exp(self.w[17] * (rating_int as f64 - 3.0 + self.w[18]))
            * stability.powf(-self.w[19]);
        // For non-Again ratings, the short-term update must never reduce stability:
        // pressing Hard/Good/Easy should not move the card "backwards".
        if rating_int >= 2 {
            sinc = sinc.max(1.0);
        }
        (stability * sinc).clamp(STABILITY_MIN, STABILITY_MAX)
    }

    pub fn next_recall_stability(
        &self,
        difficulty: f64,
        stability: f64,
        retrievability: f64,
        rating: Rating,
    ) -> f64 {
        let hard_penalty = if matches!(rating, Rating::Hard) {
            self.w[15]
        } else {
            1.0
        };
        let easy_bonus = if matches!(rating, Rating::Easy) {
            self.w[16]
        } else {
            1.0
        };
        let new_s = stability
            * (1.0
                + f64::exp(self.w[8])
                    * (11.0 - difficulty)
                    * stability.powf(-self.w[9])
                    * (f64::exp((1.0 - retrievability) * self.w[10]) - 1.0)
                    * hard_penalty
                    * easy_bonus);
        new_s.clamp(STABILITY_MIN, STABILITY_MAX)
    }

    pub fn next_forget_stability(
        &self,
        difficulty: f64,
        stability: f64,
        retrievability: f64,
    ) -> f64 {
        let new_s = self.w[11]
            * difficulty.powf(-self.w[12])
            * ((stability + 1.0).powf(self.w[13]) - 1.0)
            * f64::exp((1.0 - retrievability) * self.w[14]);
        // FSRS-6 cap: post-failure stability is capped at last_s / exp(w17 * w18),
        // so a lapse always reduces stability by at least that factor.
        let new_s_min = stability / f64::exp(self.w[17] * self.w[18]);
        new_s.min(new_s_min).clamp(STABILITY_MIN, STABILITY_MAX)
    }

    fn mean_reversion(&self, initial: f64, current: f64) -> f64 {
        self.w[7].mul_add(initial, (1.0 - self.w[7]) * current)
    }

    fn apply_fuzz(&self, interval: f64, elapsed_days: i64) -> f64 {
        if !self.enable_fuzz || interval < 2.5 {
            return interval;
        }

        let mut generator = alea(self.seed.clone());
        let fuzz_factor = generator.double();
        let (min_interval, max_interval) =
            FuzzRange::get_fuzz_range(interval, elapsed_days, self.maximum_interval);

        fuzz_factor.mul_add(
            max_interval as f64 - min_interval as f64 + 1.0,
            min_interval as f64,
        )
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            request_retention: 0.9,
            maximum_interval: 36500,
            w: DEFAULT_WEIGHTS,
            enable_short_term: true,
            enable_fuzz: false,
            seed: Seed::default(),
        }
    }
}

struct FuzzRange {
    start: f64,
    end: f64,
    factor: f64,
}

impl FuzzRange {
    const fn new(start: f64, end: f64, factor: f64) -> Self {
        Self { start, end, factor }
    }

    fn get_fuzz_range(interval: f64, elapsed_days: i64, maximum_interval: i32) -> (i64, i64) {
        let mut delta: f64 = 1.0;
        for fuzz_range in FUZZ_RANGE {
            delta += fuzz_range.factor
                * f64::max(f64::min(interval, fuzz_range.end) - fuzz_range.start, 0.0);
        }

        let i = f64::min(interval, maximum_interval as f64);
        let mut min_interval = f64::max(2.0, f64::round(i - delta));
        let max_interval: f64 = f64::min(f64::round(i + delta), maximum_interval as f64);

        if i > elapsed_days as f64 {
            min_interval = f64::max(min_interval, elapsed_days as f64 + 1.0);
        }

        min_interval = f64::min(min_interval, max_interval);

        (min_interval as i64, max_interval as i64)
    }
}

const FUZZ_RANGE: [FuzzRange; 3] = [
    FuzzRange::new(2.5, 7.0, 0.15),
    FuzzRange::new(7.0, 20.0, 0.1),
    FuzzRange::new(20.0, f64::MAX, 0.05),
];

#[derive(Debug, Clone)]
pub enum Seed {
    String(String),
    Empty,
    Default,
}

impl Seed {
    pub fn new<T>(value: T) -> Self
    where
        T: std::fmt::Display,
    {
        if value.to_string().is_empty() {
            Self::default()
        } else {
            Self::String(value.to_string())
        }
    }

    pub fn inner_str(&self) -> &str {
        match self {
            Self::String(str) => str,
            Self::Empty => Self::Default.inner_str(),
            Self::Default => Self::Default.inner_str(),
        }
    }
}

impl std::fmt::Display for Seed {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.inner_str())
    }
}

impl From<&Seed> for String {
    fn from(d: &Seed) -> Self {
        d.inner_str().to_string()
    }
}

impl From<i32> for Seed {
    fn from(num: i32) -> Self {
        Self::String(num.to_string())
    }
}

impl From<String> for Seed {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl<'a> From<&'a str> for Seed {
    fn from(s: &'a str) -> Self {
        Self::String(s.to_string())
    }
}

impl Default for Seed {
    fn default() -> Self {
        Self::String(Utc::now().timestamp_millis().to_string())
    }
}
