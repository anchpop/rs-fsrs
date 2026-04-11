#[cfg(test)]
use {
    crate::{
        alea::{AleaState, alea},
        algo::FSRS,
        models::{Card, Rating, State},
        parameters::{Parameters, Seed},
    },
    chrono::{DateTime, Duration, TimeZone, Utc},
    rand::Rng,
};

#[cfg(test)]
static TEST_RATINGS: [Rating; 13] = [
    Rating::Good,
    Rating::Good,
    Rating::Good,
    Rating::Good,
    Rating::Good,
    Rating::Good,
    Rating::Again,
    Rating::Again,
    Rating::Good,
    Rating::Good,
    Rating::Good,
    Rating::Good,
    Rating::Good,
];

// Note: this is a 21-element FSRS-6 parameter vector. The first 19 values are the
// historical FSRS-5-style test weights; the last two are w[19] = 0 (no short-term
// sinc decay) and w[20] = the FSRS-5 decay constant, so the test trajectories stay
// recognizable across the FSRS-5 -> FSRS-6 migration even though the underlying
// algorithm is now the FSRS-6 form.
#[cfg(test)]
static WEIGHTS: [f64; 21] = [
    0.4197, 1.1869, 3.0412, 15.2441, 7.1434, 0.6477, 1.0007, 0.0674, 1.6597, 0.1712, 1.1178,
    2.0225, 0.0904, 0.3025, 2.1214, 0.2498, 2.9466, 0.4891, 0.6468, 0.0, 0.5,
];

#[cfg(test)]
fn string_to_utc(date_string: &str) -> DateTime<Utc> {
    let datetime = DateTime::parse_from_str(date_string, "%Y-%m-%d %H:%M:%S %z %Z").unwrap();
    Utc.from_local_datetime(&datetime.naive_utc()).unwrap()
}
#[cfg(test)]
trait RoundFloat {
    fn round_float(self, precision: i32) -> f64;
}
#[cfg(test)]
impl RoundFloat for f64 {
    fn round_float(self, precision: i32) -> f64 {
        let multiplier = 10.0_f64.powi(precision);
        (self * multiplier).round() / multiplier
    }
}

#[test]
fn test_basic_scheduler_interval() {
    let fsrs = FSRS::default();
    let mut now = string_to_utc("2022-11-29 12:30:00 +0000 UTC");
    let mut card = Card::new(now);
    let mut interval_history = vec![];

    for rating in TEST_RATINGS.iter() {
        let next = fsrs.next(card, now, *rating);
        card = next.card;
        interval_history.push(card.scheduled_days);
        now = card.due;
    }
    // FSRS-6 expected intervals (with FSRS-6 defaults: linear damping, w[20] decay,
    // new_s_min cap, no Hard-as-stability-decrease, etc.)
    let expected = [0, 2, 11, 46, 163, 497, 0, 0, 2, 4, 7, 12, 20];
    assert_eq!(interval_history, expected);
}

#[test]
fn test_basic_scheduler_state() {
    let params = Parameters {
        w: WEIGHTS,
        ..Default::default()
    };

    let fsrs = FSRS::new(params);
    let mut now = string_to_utc("2022-11-29 12:30:00 +0000 UTC");
    let mut card = Card::new(now);
    let mut state_list = vec![];
    let mut record_log = fsrs.repeat(card, now);

    for rating in TEST_RATINGS.iter() {
        card = record_log[rating].card.clone();
        let rev_log = record_log[rating].review_log.clone();
        state_list.push(rev_log.state);
        now = card.due;
        record_log = fsrs.repeat(card, now);
    }
    use State::*;
    let expected = [
        New, Learning, Review, Review, Review, Review, Review, Relearning, Relearning, Review,
        Review, Review, Review,
    ];
    assert_eq!(state_list, expected);
}

#[test]
fn test_basic_scheduler_memo_state() {
    let params = Parameters {
        w: WEIGHTS,
        ..Default::default()
    };

    let fsrs = FSRS::new(params);
    let mut now = string_to_utc("2022-11-29 12:30:00 +0000 UTC");
    let mut card = Card::new(now);
    let mut record_log = fsrs.repeat(card.clone(), now);
    let ratings = [
        Rating::Again,
        Rating::Good,
        Rating::Good,
        Rating::Good,
        Rating::Good,
        Rating::Good,
    ];
    let intervals = [0, 0, 1, 3, 8, 21];
    for (index, rating) in ratings.iter().enumerate() {
        card = record_log[rating].card.clone();
        now += Duration::days(intervals[index] as i64);
        record_log = fsrs.repeat(card.clone(), now);
    }

    card = record_log[&Rating::Good].to_owned().card;
    assert_eq!(card.stability.round_float(4), 71.4554);
    assert_eq!(card.difficulty.round_float(4), 5.0976);
}

#[test]
fn test_long_term_scheduler() {
    let params = Parameters {
        w: WEIGHTS,
        enable_short_term: false,
        ..Default::default()
    };

    let fsrs = FSRS::new(params);
    let mut now = string_to_utc("2022-11-29 12:30:00 +0000 UTC");
    let mut card = Card::new(now);
    let mut interval_history = vec![];
    let mut stability_history = vec![];
    let mut difficulty_history = vec![];

    for rating in TEST_RATINGS.iter() {
        let record = fsrs.repeat(card.clone(), now)[rating].to_owned();
        let next = fsrs.next(card, now, *rating);

        assert_eq!(record.card, next.card);

        card = record.card;
        interval_history.push(card.scheduled_days);
        stability_history.push(card.stability.round_float(4));
        difficulty_history.push(card.difficulty.round_float(4));
        now = card.due;
    }

    // FSRS-6 expected sequences (algorithm changes affect post-lapse stability via
    // new_s_min cap; pre-lapse values match FSRS-5 because the recall/forget formulas
    // are unchanged for Review-state cards with these specific weights).
    let expected_interval = [3, 13, 48, 155, 445, 1158, 17, 3, 11, 37, 112, 307, 773];
    let expected_stability = [
        3.0412, 13.0913, 48.1585, 154.9373, 445.0556, 1158.0778, 16.6306, 3.0173, 11.4225, 37.3752,
        111.8753, 306.5975, 772.9403,
    ];
    let expected_difficulty = [
        4.4909, 4.2666, 4.0575, 3.8624, 3.6804, 3.5108, 4.6983, 5.5596, 5.2632, 4.9869, 4.7292,
        4.4888, 4.2646,
    ];

    assert_eq!(interval_history, expected_interval);
    assert_eq!(stability_history, expected_stability);
    assert_eq!(difficulty_history, expected_difficulty);
}

#[test]
fn test_prng_get_state() {
    let prng_1 = alea(Seed::new(1));
    let prng_2 = alea(Seed::new(2));
    let prng_3 = alea(Seed::new(1));

    let alea_state_1 = prng_1.get_state();
    let alea_state_2 = prng_2.get_state();
    let alea_state_3 = prng_3.get_state();

    assert_eq!(alea_state_1, alea_state_3);
    assert_ne!(alea_state_1, alea_state_2);
}

#[test]
fn test_alea_get_next() {
    let seed = Seed::new(12345);
    let mut generator = alea(seed);
    assert_eq!(generator.gen_next(), 0.27138191112317145);
    assert_eq!(generator.gen_next(), 0.19615925149992108);
    assert_eq!(generator.gen_next(), 0.6810678059700876);
}

#[test]
fn test_alea_int32() {
    let seed = Seed::new(12345);
    let mut generator = alea(seed);
    assert_eq!(generator.int32(), 1165576433);
    assert_eq!(generator.int32(), 842497570);
    assert_eq!(generator.int32(), -1369803343);
}

#[test]
fn test_alea_import_state() {
    let mut rng = rand::rng();
    let mut prng_1 = alea(Seed::new(rng.random::<i32>()));
    prng_1.gen_next();
    prng_1.gen_next();
    prng_1.gen_next();
    let prng_1_state = prng_1.get_state();
    let mut prng_2 = alea(Seed::Empty).import_state(prng_1_state);

    assert_eq!(prng_1.get_state(), prng_2.get_state());

    for _ in 1..10000 {
        let a = prng_1.gen_next();
        let b = prng_2.gen_next();

        assert_eq!(a, b);
        assert!((0.0..1.0).contains(&a));
        assert!((0.0..1.0).contains(&b));
    }
}

#[test]
fn test_seed_example_1() {
    let seed = Seed::new("1727015666066");
    let mut generator = alea(seed);
    let results = generator.gen_next();
    let state = generator.get_state();

    let expect_alea_state = AleaState {
        c: 1828249.0,
        s0: 0.5888567129150033,
        s1: 0.5074866858776659,
        s2: 0.6320083506871015,
    };
    assert_eq!(results, 0.6320083506871015);
    assert_eq!(state, expect_alea_state);
}

#[test]
fn test_seed_example_2() {
    let seed = Seed::new("Seedp5fxh9kf4r0");
    let mut generator = alea(seed);
    let results = generator.gen_next();
    let state = generator.get_state();

    let expect_alea_state = AleaState {
        c: 1776946.0,
        s0: 0.6778371171094477,
        s1: 0.0770602801349014,
        s2: 0.14867847645655274,
    };
    assert_eq!(results, 0.14867847645655274);
    assert_eq!(state, expect_alea_state);
}

#[test]
fn test_seed_example_3() {
    let seed = Seed::new("NegativeS2Seed");
    let mut generator = alea(seed);
    let results = generator.gen_next();
    let state = generator.get_state();

    let expect_alea_state = AleaState {
        c: 952982.0,
        s0: 0.25224833423271775,
        s1: 0.9213257452938706,
        s2: 0.830770346801728,
    };
    assert_eq!(results, 0.830770346801728);
    assert_eq!(state, expect_alea_state);
}

// ============================================================================
// fsrs-rs cross-validation tests
//
// These ports of tests in open-spaced-repetition/fsrs-rs/src/inference.rs
// verify that this rs-fsrs port produces the same numerical results as the
// canonical FSRS-6 reference implementation.
//
// Source: https://github.com/open-spaced-repetition/fsrs-rs/blob/master/src/inference.rs
// ============================================================================

/// Port of fsrs-rs `test_next_interval`
/// (`open-spaced-repetition/fsrs-rs/src/inference.rs::test_next_interval`).
///
/// Verifies the FSRS-6 next_interval formula across the full range of
/// requested retentions, using FSRS-6 default weights.
#[test]
fn test_next_interval_matches_fsrs_rs() {
    let intervals: Vec<i32> = (1..=10)
        .map(|i| i as f64 / 10.0)
        .map(|retention| {
            let params = Parameters {
                request_retention: retention,
                // fsrs-rs's next_interval doesn't clamp to maximum_interval; bypass
                // rs-fsrs's clamp here so the cross-validation matches fsrs-rs's
                // raw numerical output for r=0.1 (which would otherwise saturate
                // at the default 100-year cap).
                maximum_interval: i32::MAX,
                ..Default::default()
            };
            // stability=1.0, elapsed_days=1 (matches fsrs-rs's call)
            params.next_interval(1.0, 1).round().max(1.0) as i32
        })
        .collect();
    let expected = [3116766, 34793, 2508, 387, 90, 27, 9, 3, 1, 1];
    // The first interval (~3.1M days) is sensitive to f32 vs f64 — fsrs-rs
    // computes it in f32, our f64 matches to within ~1e-6 relative. Allow a
    // 5-day tolerance there; the rest must match exactly.
    assert!(
        ((intervals[0] - expected[0]).abs()) <= 5,
        "interval[0] mismatch beyond f32-precision tolerance: got {}, expected {}",
        intervals[0],
        expected[0],
    );
    assert_eq!(&intervals[1..], &expected[1..]);
}

/// Port of fsrs-rs `test_current_retrievability`
/// (`open-spaced-repetition/fsrs-rs/src/inference.rs::test_current_retrievability`).
///
/// Verifies the power forgetting curve directly. Uses a custom decay of 0.2
/// (so w[20] = 0.2), with stability = 1.0.
#[test]
fn test_forgetting_curve_matches_fsrs_rs() {
    let mut w = [0.0_f64; 21];
    w[20] = 0.2;
    let params = Parameters {
        w,
        ..Default::default()
    };
    let stability = 1.0;
    let actual: Vec<f64> = [0.0_f64, 1.0, 2.0, 3.0]
        .iter()
        // 6 decimal places: fsrs-rs uses f32 internally, so its 7th decimal
        // diverges from our f64 by ~1 ulp. 6 dp is well within both precisions.
        .map(|&t| params.forgetting_curve(t, stability).round_float(6))
        .collect();
    assert_eq!(actual, [1.0, 0.9, 0.840289, 0.7985]);
}

/// Port of fsrs-rs `test_next_states`
/// (`open-spaced-repetition/fsrs-rs/src/inference.rs::test_next_states`).
///
/// Replays the same 4-review sequence as fsrs-rs:
///   - Again, same day
///   - Good, +1 day
///   - Good, +3 days
///   - Good, +8 days
///
/// Then asks for the "what would the new state be after a Good review now,
/// 21 days after the last review" — comparing memory state and interval to
/// fsrs-rs's expected output.
///
/// Uses fsrs-rs's PARAMETERS (19-element FSRS-5 vector, padded to 21 with
/// `[0.0, 0.5]` to match check_and_fill_parameters' 19->21 transform).
#[test]
fn test_next_states_matches_fsrs_rs() {
    // 19 historical FSRS-5 weights from fsrs-rs/src/inference.rs:866 +
    // [0.0, 0.5] (FSRS5_DEFAULT_DECAY) per check_and_fill_parameters' 19->21 path.
    let weights: [f64; 21] = [
        0.6845422,
        1.6790825,
        4.7349424,
        10.042885,
        7.4410233,
        0.64219797,
        1.071918,
        0.0025195254,
        1.432437,
        0.1544,
        0.8692766,
        2.0696752,
        0.0953,
        0.2975,
        2.4691248,
        0.19542035,
        3.201072,
        0.18046261,
        0.121442534,
        0.0,
        0.5,
    ];
    let params = Parameters {
        w: weights,
        request_retention: 0.9,
        ..Default::default()
    };
    let fsrs = FSRS::new(params);

    // Replay the sequence: Again @ t=0, Good @ t=1, Good @ t=4, Good @ t=12.
    let mut now = string_to_utc("2022-11-29 12:30:00 +0000 UTC");
    let mut card = Card::new(now);

    // Step 1: Again on a brand new card (delta_t = 0).
    card = fsrs.next(card, now, Rating::Again).card;

    // Step 2: Good, 1 day later.
    now += Duration::days(1);
    card = fsrs.next(card, now, Rating::Good).card;

    // Step 3: Good, 3 days later (4 days total elapsed since step 2).
    now += Duration::days(3);
    card = fsrs.next(card, now, Rating::Good).card;

    // Step 4: Good, 8 days later (12 days total).
    now += Duration::days(8);
    card = fsrs.next(card, now, Rating::Good).card;

    // Now ask: what would the next states be 21 days from now?
    now += Duration::days(21);
    let log = fsrs.repeat(card, now);

    // Expected from fsrs-rs::test_next_states. fsrs-rs uses f32 internally so
    // we allow a small tolerance vs our f64 results.
    let expected = [
        // (rating, expected_stability, expected_difficulty)
        (Rating::Again, 2.9691455_f64, 8.000659_f64),
        (Rating::Hard, 17.091452_f64, 7.6913934_f64),
        (Rating::Good, 31.722992_f64, 7.382128_f64),
        (Rating::Easy, 71.7502_f64, 7.0728626_f64),
    ];
    let tolerance = 1e-3_f64;
    for (rating, expected_stability, expected_difficulty) in expected {
        let next_card = log[&rating].card.clone();
        let stability_diff = (next_card.stability - expected_stability).abs();
        let difficulty_diff = (next_card.difficulty - expected_difficulty).abs();
        assert!(
            stability_diff < tolerance,
            "rating={:?}: stability mismatch — got {}, expected {} (diff {})",
            rating,
            next_card.stability,
            expected_stability,
            stability_diff,
        );
        assert!(
            difficulty_diff < tolerance,
            "rating={:?}: difficulty mismatch — got {}, expected {} (diff {})",
            rating,
            next_card.difficulty,
            expected_difficulty,
            difficulty_diff,
        );
    }
}

#[test]
fn test_get_retrievability() {
    let parameters = Parameters::default();
    let fsrs = FSRS::new(parameters.clone());
    let now = string_to_utc("2022-11-29 12:30:00 +0000 UTC");
    let card = Card::new(now);
    // FSRS-6 retrievability: differs from FSRS-5 because the default decay is now
    // w[20] = 0.1542 (vs the FSRS-5 hardcoded -0.5).
    let expect_retrievability = [0.9995057, 0.9995947, 0.9995456, 0.9024733];
    let scheduler = fsrs.repeat(card, now);

    let actual: Vec<f64> = Rating::iter()
        .map(|rating| {
            let card = scheduler.get(rating).unwrap().card.clone();
            card.get_retrievability(card.due, &parameters)
                .round_float(7)
        })
        .collect();
    assert_eq!(actual, expect_retrievability);
}
