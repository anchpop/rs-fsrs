use crate::Parameters;
use chrono::{DateTime, Utc};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Debug, Default, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum State {
    #[default]
    New = 0,
    Learning = 1,
    Review = 2,
    Relearning = 3,
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Rating {
    Again = 1,
    Hard = 2,
    Good = 3,
    Easy = 4,
}

impl Rating {
    pub fn iter() -> std::slice::Iter<'static, Self> {
        static VARIANTS: [Rating; 4] = [Rating::Again, Rating::Hard, Rating::Good, Rating::Easy];
        VARIANTS.iter()
    }
}

#[derive(Debug, Clone)]
pub struct SchedulingInfo {
    pub card: Card,
    pub review_log: ReviewLog,
}

pub type RecordLog = HashMap<Rating, SchedulingInfo>;

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReviewLog {
    pub rating: Rating,
    pub elapsed_days: i64,
    pub scheduled_days: i64,
    pub state: State,
    pub reviewed_date: DateTime<Utc>,
}

#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Card {
    pub due: DateTime<Utc>,
    pub stability: f64,
    pub difficulty: f64,
    pub elapsed_days: i64,
    pub scheduled_days: i64,
    pub reps: i32,
    pub lapses: i32,
    pub state: State,
    pub last_review: DateTime<Utc>,
    pub accumulated_positive_surprise: f64,
    pub accumulated_negative_surprise: f64,
}

impl Card {
    pub fn new() -> Self {
        Self {
            due: Utc::now(),
            last_review: Utc::now(),
            accumulated_positive_surprise: 0.0,
            accumulated_negative_surprise: 0.0,
            ..Default::default()
        }
    }

    pub fn get_retrievability(&self, now: DateTime<Utc>) -> f64 {
        match self.state {
            State::New => 0.0,
            _ => {
                let elapsed_days = now.signed_duration_since(self.last_review).num_days();
                Parameters::forgetting_curve(elapsed_days as f64, self.stability)
            }
        }
    }

    /// Updates card metrics that are common across all schedulers
    /// This includes lapses and surprise accumulation
    pub(crate) fn update_metrics(
        &mut self,
        rating: Rating,
        retrievability: f64,
        current_card: &Card,
    ) {
        // Update lapses for failed reviews
        if rating == Rating::Again {
            self.lapses = current_card.lapses + 1;
        } else {
            self.lapses = current_card.lapses;
        }

        // Calculate probability of the observed outcome
        let probability = if retrievability == 0.0 {
            // For new cards or when retrievability is 0, use 0.5 as default
            0.1
        } else {
            retrievability
        };

        let probability = if rating == Rating::Again {
            // Failed review: probability of failure is (1 - retrievability)
            1.0 - probability
        } else {
            // Successful review: probability of success is retrievability
            probability
        };

        // Convert probability to surprise using information theory: -log(probability)
        // Clamp to avoid log(0) for numerical stability
        let mut surprise = -(probability.max(0.001).ln());

        // Adjust surprise for different positive rating levels
        match rating {
            Rating::Hard => surprise *= 0.5, // Hard is less surprising
            Rating::Easy => surprise *= 2.0, // Easy is more surprising
            _ => {}                          // Good and Again remain unchanged
        }

        // Accumulate surprise based on rating
        if rating == Rating::Again {
            self.accumulated_negative_surprise =
                current_card.accumulated_negative_surprise + surprise;
            self.accumulated_positive_surprise = current_card.accumulated_positive_surprise;
        } else {
            self.accumulated_positive_surprise =
                current_card.accumulated_positive_surprise + surprise;
            self.accumulated_negative_surprise = current_card.accumulated_negative_surprise;
        }
    }
}
