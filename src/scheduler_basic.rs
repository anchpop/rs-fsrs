use chrono::{DateTime, Duration, Utc};

use crate::{Card, ImplScheduler, Parameters, Rating, SchedulingInfo, scheduler::Scheduler};
use crate::{Rating::*, State::*};
pub struct BasicScheduler {
    pub scheduler: Scheduler,
}

impl BasicScheduler {
    pub fn new(parameters: Parameters, card: Card, now: DateTime<Utc>) -> Self {
        Self {
            scheduler: Scheduler::new(parameters, card, now),
        }
    }
    fn new_state(&mut self, rating: Rating) -> SchedulingInfo {
        if let Some(exist) = self.scheduler.next.get(&rating) {
            return exist.clone();
        }

        let mut next = self.scheduler.current.clone();
        next.difficulty = self.scheduler.parameters.init_difficulty(rating);
        next.stability = self.scheduler.parameters.init_stability(rating);

        match rating {
            Again => {
                next.scheduled_days = 0;
                next.due = self.scheduler.now + Duration::minutes(1);
                next.state = Learning;
            }
            Hard => {
                next.scheduled_days = 0;
                next.due = self.scheduler.now + Duration::minutes(3);
                next.state = Learning;
            }
            Good => {
                next.scheduled_days = 0;
                next.due = self.scheduler.now + Duration::minutes(6);
                next.state = Learning;
            }
            Easy => {
                let easy_interval = self
                    .scheduler
                    .parameters
                    .next_interval(next.stability, next.elapsed_days);
                next.scheduled_days = easy_interval as i64;
                next.due = self.scheduler.now + Duration::days(easy_interval as i64);
                next.state = Review;
            }
        };

        // Update common metrics for new cards (retrievability is 0 for new cards)
        next.update_metrics(rating, 0.0, &self.scheduler.current);

        let item = SchedulingInfo {
            card: next,
            review_log: self.scheduler.build_log(rating),
        };

        self.scheduler.next.insert(rating, item.clone());
        item
    }

    fn learning_state(&mut self, rating: Rating) -> SchedulingInfo {
        if let Some(exist) = self.scheduler.next.get(&rating) {
            return exist.clone();
        }

        let mut next = self.scheduler.current.clone();
        let interval = self.scheduler.current.elapsed_days;
        let last_stability = self.scheduler.last.stability;
        let last_difficulty = self.scheduler.last.difficulty;
        let retrievability = self
            .scheduler
            .last
            .get_retrievability(self.scheduler.now, &self.scheduler.parameters);

        next.difficulty = self
            .scheduler
            .parameters
            .next_difficulty(last_difficulty, rating);

        // FSRS-6 gate: when at least one day has elapsed since the last review,
        // use the proper retrievability-aware update. Same-day reviews still use
        // the short-term update so in-batch drilling produces meaningful changes
        // (the forgetting curve has barely moved over seconds, so a "Good" via
        // the long-term path would change stability by ~0).
        next.stability = if interval > 0 {
            match rating {
                Again => self.scheduler.parameters.next_forget_stability(
                    last_difficulty,
                    last_stability,
                    retrievability,
                ),
                _ => self.scheduler.parameters.next_recall_stability(
                    last_difficulty,
                    last_stability,
                    retrievability,
                    rating,
                ),
            }
        } else {
            self.scheduler
                .parameters
                .short_term_stability(last_stability, rating)
        };

        match rating {
            Again => {
                next.scheduled_days = 0;
                next.due = self.scheduler.now + Duration::minutes(5);
                next.state = self.scheduler.last.state;
            }
            Hard => {
                next.scheduled_days = 0;
                next.due = self.scheduler.now + Duration::minutes(6);
                next.state = self.scheduler.last.state;
            }
            Good => {
                let good_interval = self
                    .scheduler
                    .parameters
                    .next_interval(next.stability, interval);
                next.scheduled_days = good_interval as i64;
                next.due = self.scheduler.now + Duration::days(good_interval as i64);
                next.state = Review;
            }
            Easy => {
                // Compute a "good_stability" comparable to next.stability so the
                // Easy interval is strictly longer than the Good interval.
                let good_stability = if interval > 0 {
                    self.scheduler.parameters.next_recall_stability(
                        last_difficulty,
                        last_stability,
                        retrievability,
                        Good,
                    )
                } else {
                    self.scheduler
                        .parameters
                        .short_term_stability(last_stability, Good)
                };
                let good_interval = self
                    .scheduler
                    .parameters
                    .next_interval(good_stability, interval);
                let easy_interval = self
                    .scheduler
                    .parameters
                    .next_interval(next.stability, interval)
                    .max(good_interval + 1.0);
                next.scheduled_days = easy_interval as i64;
                next.due = self.scheduler.now + Duration::days(easy_interval as i64);
                next.state = Review;
            }
        }

        // Update common metrics for learning/relearning cards
        next.update_metrics(rating, retrievability, &self.scheduler.current);

        let item = SchedulingInfo {
            card: next,
            review_log: self.scheduler.build_log(rating),
        };

        self.scheduler.next.insert(rating, item.clone());
        item
    }

    fn review_state(&mut self, rating: Rating) -> SchedulingInfo {
        if let Some(exist) = self.scheduler.next.get(&rating) {
            return exist.clone();
        }

        let next = self.scheduler.current.clone();
        let interval = self.scheduler.current.elapsed_days;
        let stability = self.scheduler.last.stability;
        let difficulty = self.scheduler.last.difficulty;
        let retrievability = self
            .scheduler
            .last
            .get_retrievability(self.scheduler.now, &self.scheduler.parameters);

        let mut next_again = next.clone();
        let mut next_hard = next.clone();
        let mut next_good = next.clone();
        let mut next_easy = next;

        self.next_difficulty_stability(
            &mut next_again,
            &mut next_hard,
            &mut next_good,
            &mut next_easy,
            interval,
            difficulty,
            stability,
            retrievability,
        );
        self.next_interval(
            &mut next_again,
            &mut next_hard,
            &mut next_good,
            &mut next_easy,
            interval,
        );
        self.next_state(
            &mut next_again,
            &mut next_hard,
            &mut next_good,
            &mut next_easy,
        );

        // Update common metrics (lapses and surprise) for all rating options
        next_again.update_metrics(Rating::Again, retrievability, &self.scheduler.current);
        next_hard.update_metrics(Rating::Hard, retrievability, &self.scheduler.current);
        next_good.update_metrics(Rating::Good, retrievability, &self.scheduler.current);
        next_easy.update_metrics(Rating::Easy, retrievability, &self.scheduler.current);

        let item_again = SchedulingInfo {
            card: next_again,
            review_log: self.scheduler.build_log(Again),
        };
        let item_hard = SchedulingInfo {
            card: next_hard,
            review_log: self.scheduler.build_log(Hard),
        };
        let item_good = SchedulingInfo {
            card: next_good,
            review_log: self.scheduler.build_log(Good),
        };
        let item_easy = SchedulingInfo {
            card: next_easy,
            review_log: self.scheduler.build_log(Easy),
        };

        self.scheduler.next.insert(Again, item_again);
        self.scheduler.next.insert(Hard, item_hard);
        self.scheduler.next.insert(Good, item_good);
        self.scheduler.next.insert(Easy, item_easy);

        self.scheduler.next.get(&rating).unwrap().to_owned()
    }

    #[allow(clippy::too_many_arguments)]
    fn next_difficulty_stability(
        &self,
        next_again: &mut Card,
        next_hard: &mut Card,
        next_good: &mut Card,
        next_easy: &mut Card,
        elapsed_days: i64,
        difficulty: f64,
        stability: f64,
        retrievability: f64,
    ) {
        next_again.difficulty = self.scheduler.parameters.next_difficulty(difficulty, Again);
        next_again.stability = if elapsed_days > 0 {
            self.scheduler
                .parameters
                .next_forget_stability(difficulty, stability, retrievability)
        } else {
            self.scheduler
                .parameters
                .short_term_stability(stability, Again)
        };

        next_hard.difficulty = self.scheduler.parameters.next_difficulty(difficulty, Hard);
        next_hard.stability = if elapsed_days > 0 {
            self.scheduler.parameters.next_recall_stability(
                difficulty,
                stability,
                retrievability,
                Hard,
            )
        } else {
            self.scheduler
                .parameters
                .short_term_stability(stability, Hard)
        };

        next_good.difficulty = self.scheduler.parameters.next_difficulty(difficulty, Good);
        next_good.stability = if elapsed_days > 0 {
            self.scheduler.parameters.next_recall_stability(
                difficulty,
                stability,
                retrievability,
                Good,
            )
        } else {
            self.scheduler
                .parameters
                .short_term_stability(stability, Good)
        };

        next_easy.difficulty = self.scheduler.parameters.next_difficulty(difficulty, Easy);
        next_easy.stability = if elapsed_days > 0 {
            self.scheduler.parameters.next_recall_stability(
                difficulty,
                stability,
                retrievability,
                Easy,
            )
        } else {
            self.scheduler
                .parameters
                .short_term_stability(stability, Easy)
        };
    }

    fn next_interval(
        &self,
        next_again: &mut Card,
        next_hard: &mut Card,
        next_good: &mut Card,
        next_easy: &mut Card,
        elapsed_days: i64,
    ) {
        let mut hard_interval = self
            .scheduler
            .parameters
            .next_interval(next_hard.stability, elapsed_days);
        let mut good_interval = self
            .scheduler
            .parameters
            .next_interval(next_good.stability, elapsed_days);
        hard_interval = hard_interval.min(good_interval);
        good_interval = good_interval.max(hard_interval + 1.0);
        let easy_interval = self
            .scheduler
            .parameters
            .next_interval(next_easy.stability, elapsed_days)
            .max(good_interval + 1.0);

        next_again.scheduled_days = 0;
        next_again.due = self.scheduler.now + Duration::minutes(5);

        next_hard.scheduled_days = hard_interval as i64;
        next_hard.due = self.scheduler.now + Duration::days(hard_interval as i64);

        next_good.scheduled_days = good_interval as i64;
        next_good.due = self.scheduler.now + Duration::days(good_interval as i64);

        next_easy.scheduled_days = easy_interval as i64;
        next_easy.due = self.scheduler.now + Duration::days(easy_interval as i64);
    }

    fn next_state(
        &self,
        next_again: &mut Card,
        next_hard: &mut Card,
        next_good: &mut Card,
        next_easy: &mut Card,
    ) {
        next_again.state = Relearning;
        next_hard.state = Review;
        next_good.state = Review;
        next_easy.state = Review;
    }
}

impl ImplScheduler for BasicScheduler {
    fn review(&mut self, rating: Rating) -> SchedulingInfo {
        match self.scheduler.last.state {
            New => self.new_state(rating),
            Learning | Relearning => self.learning_state(rating),
            Review => self.review_state(rating),
        }
    }
}
