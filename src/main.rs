/*
NN Step 15. Train One Neuron
[src/solutions/nn/15-TrainOneNeuron.rs]

Goal:
Train a single neuron on a tiny dataset using gradient descent.

Why this matters for neural networks:
Up to now, you've implemented building blocks:
- Forward pass for a neuron/layer
- Loss computation
- Backprop gradients (dW, db)
- An SGD parameter update step (Step 14)

But models don't learn from a single gradient step.
You need to loop:
  forward -> loss -> backward -> update
over many examples and epochs.

This step wires all the pieces into a training loop
for a single neuron, so you see end-to-end learning
on a simple task.

Setup:
We consider a single neuron with:
- weights w: Vec<f32>           (one weight per input dimension)
- bias   b: f32
- activation: often identity or sigmoid (check TODOs/tests)
- a small dataset of (x, y) pairs.

Forward rule (linear neuron):
  z = sum_k w[k] * x[k] + b
  y_hat = z             (if no activation)
or:
  y_hat = sigmoid(z)    (if specified)

Loss (for this exercise, typically MSE):
  loss = 0.5 * (y_hat - y)^2

Gradients (for linear + MSE):
  dL/dy_hat = (y_hat - y)
  dL/dw[k]  = dL/dy_hat * x[k]
  dL/db     = dL/dy_hat

Training loop:
For each epoch:
  - Initialize accumulators for dW and db (all zeros)
  - For each sample (x, y):
      * compute y_hat
      * accumulate dW and db
  - After the dataset, apply one SGD update:
      w <- w - lr * dW / N
      b <- b - lr * db / N

Notes:
- The exact averaging / scaling might be in the TODOs.
- Follow the step comments exactly; tests are sensitive.
*/


struct Solution;

// A simple neuron: linear combination + optional activation
pub struct Neuron {
    pub w: Vec<f32>, // weights
    pub b: f32,      // bias
}


/// Hints:
/// 1) Think: z = w·x + b.
/// 2) Inputs x and weights w must have same length.
/// 3) Fold over indices or use iter().zip().
impl Neuron {
    pub fn forward(&self, x: &[f32]) -> f32 {
        // TODO: implement forward pass for one input x

        // HINT 1: Compute dot product between self.w and x, then add self.b.
        // HINT 2: z = sum_{k} self.w[k] * x[k]; return z (if no activation).
        // HINT 3: 
        // let mut z = 0.0;
        // for k in 0..self.w.len() {
        //     z += self.w[k] * x[k];
        // }
        // z + self.b

        let cols = x.len();

        let mut z = 0.0;
        for k in 0..cols {
            z += x[k] * self.w[k];
        }

        z + self.b        
    }
}


/// Hints:
/// 1) Use MSE-like loss: 0.5 * (y_hat - y)^2.
/// 2) This keeps gradients simple: derivative wrt y_hat is (y_hat - y).
/// 3) You’ve done this earlier when implementing scalar losses.
pub fn loss_mse(y_hat: f32, y: f32) -> f32 {
    // TODO: implement mean-squared-error (scalar)

    // HINT 1: error = y_hat - y.
    // HINT 2: loss = 0.5 * error * error.
    // HINT 3:
    // let e = y_hat - y;
    // 0.5 * e * e

    let error = y_hat - y;
    0.5 * (error) * (error)
}


/// Hints:
/// 1) Compute gradients for a single sample (x, y).
/// 2) Start from dL/dy_hat = (y_hat - y).
/// 3) Then dL/dw[k] = dL/dy_hat * x[k], dL/db = dL/dy_hat.
pub fn neuron_gradients(
    neuron: &Neuron,
    x: &[f32],
    y: f32,
) -> (Vec<f32>, f32) {
    // TODO: implement per-sample gradients dW, db

    // HINT 1: Forward: let y_hat = neuron.forward(x).
    // HINT 2: let diff = y_hat - y;
    // HINT 3:
    // let mut dW = vec![0.0; neuron.w.len()];
    // for k in 0..neuron.w.len() {
    //     dW[k] = diff * x[k];
    // }
    // let db = diff;
    let y_hat = neuron.forward(x);
    let diff = y_hat - y;
    let mut dW = vec![0.0; neuron.w.len()];
    for k in 0..neuron.w.len() {
        dW[k] = diff * x[k];
    }

    let db = diff;

    (dW, db)
}


/// Hints:
/// 1) This is basically Step 14 but for a single neuron.
/// 2) You can either inline the vector update or reuse your Step 14 logic.
/// 3) Think: w[k] -= lr * dW[k]; b -= lr * db.
pub fn sgd_update_neuron(
    neuron: &mut Neuron,
    dW: &[f32],
    db: f32,
    lr: f32,
) {
    // TODO: implement SGD update for a single neuron

    // HINT 1: Loop over weights: neuron.w[k] = neuron.w[k] - lr * dW[k].
    // HINT 2: Update bias: neuron.b = neuron.b - lr * db.
    // HINT 3: Make sure dW.len() == neuron.w.len() (tests should ensure this).

    for k in 0..neuron.w.len() {
        neuron.w[k] = neuron.w[k] - lr * dW[k];
    }

    neuron.b = neuron.b - lr * db;
}


/// Hints:
/// 1) This is the core training loop for Step 15.
/// 2) You will:
///    - Initialize a Neuron (w, b)
///    - Loop over epochs
///    - For each epoch:
///        * accumulate gradients over all samples
///        * average them (optional, depending on comments)
///        * call sgd_update_neuron
/// 3) Tests will likely check final weights/bias for a toy dataset.
impl Solution {
    pub fn train_one_neuron(
        xs: Vec<Vec<f32>>, // training inputs: N x D
        ys: Vec<f32>,      // training targets: length N
        lr: f32,
        epochs: usize,
    ) -> Neuron {
        // TODO: implement
        //
        // 1. Basic sanity checks (optionally):
        //    - xs.len() == ys.len()
        //    - xs non-empty, xs[0].len() is input_dim
        // [since Neurons is not of the option type, no need]

        // 2. Initialize neuron:
        //    - w = vec![0.0; input_dim]
        //    - b = 0.0
        //

        let mut neuron = Neuron {
            w: vec![0.0; xs[0].len()],
            b: 0.0,
        };

        let n = xs.len() as f32;

        for _ in 0..epochs {
            let mut db_sum = 0.0;
            let mut dW_sum = vec![0.0; neuron.w.len()];

            for i in 0..xs.len() {
                let x = &xs[i];
                let y = ys[i];

                let (dW_i, db_i) = neuron_gradients(&neuron, x, y);

                for k in 0..dW_i.len() {
                    dW_sum[k] += dW_i[k];
                }

                db_sum += db_i;
            }
            
            let mut dW_avg = vec![0.0; neuron.w.len()];
            for k in 0..neuron.w.len() {
                dW_avg[k] = dW_sum[k] / n;
            }
            let db_avg = db_sum / n;

            sgd_update_neuron(&mut neuron, &dW_avg, db_avg, lr);       

        }

        neuron
    }
}


// Tests

use std::thread;
use std::time::Duration;

const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

type Dataset = (Vec<Vec<f32>>, Vec<f32>);

struct RunConfig {
    lr: f32,
    epochs: usize,
    n_train: usize,
    n_show: usize,
    delay_ms: u64,
}

struct DemoCase {
    title: &'static str,
    tolerance: f32,
    dataset: Dataset,
}

struct DemoSummary {
    title: String,
    shown: usize,
    passed: usize,
    mae: f32,
    score: f32,
}

impl DemoSummary {
    fn score_label(&self) -> &'static str {
        if self.score >= 99.0 {
            "Excellent fit"
        } else if self.score >= 90.0 {
            "Strong fit"
        } else if self.score >= 75.0 {
            "Decent fit"
        } else if self.score >= 50.0 {
            "Weak fit"
        } else {
            "Poor fit"
        }
    }

    fn all_passed(&self) -> bool {
        self.passed == self.shown
    }
}

struct Runner;

impl Runner {
    fn print_banner(title: &str) {
        println!("{BOLD}{MAGENTA}============================={RESET}");
        println!("{BOLD}{MAGENTA}  {title}{RESET}");
        println!("{BOLD}{MAGENTA}============================={RESET}");
    }

    fn status_label(passed: bool) -> &'static str {
        if passed { "PASS" } else { "FAIL" }
    }

    fn status_color(passed: bool) -> &'static str {
        if passed { GREEN } else { RED }
    }

    fn build_linear_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        let n_f = n as f32;
        for i in 0..n {
            let t = i as f32 / (n_f - 1.0).max(1.0);
            let x = -5.0 + 10.0 * t;
            let y = 2.0 * x + 1.0;
            xs.push(vec![x]);
            ys.push(y);
        }

        (xs, ys)
    }

    fn build_exponential_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        let n_f = n as f32;
        for i in 0..n {
            let t = i as f32 / (n_f - 1.0).max(1.0);
            let x = 4.0 * t;
            let y = (0.5 * x).exp();
            xs.push(vec![x]);
            ys.push(y);
        }

        (xs, ys)
    }

    fn build_quadratic_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        let n_f = n as f32;
        for i in 0..n {
            let t = i as f32 / (n_f - 1.0).max(1.0);
            let x = -2.0 + 4.0 * t;
            let y = x * x;
            xs.push(vec![x]);
            ys.push(y);
        }

        (xs, ys)
    }

    fn build_affine_2d_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        let side = (n as f32).sqrt().floor() as usize;
        let side = side.max(2);
        let side_f = side as f32;

        for i in 0..side {
            for j in 0..side {
                if xs.len() >= n {
                    break;
                }

                let ti = i as f32 / (side_f - 1.0).max(1.0);
                let tj = j as f32 / (side_f - 1.0).max(1.0);

                let x1 = -3.0 + 6.0 * ti;
                let x2 = -3.0 + 6.0 * tj;
                let y = 3.0 * x1 - x2 + 0.5;

                xs.push(vec![x1, x2]);
                ys.push(y);
            }
        }

        (xs, ys)
    }

    fn build_cases(n_train: usize) -> Vec<DemoCase> {
        vec![
            DemoCase {
                title: "NN Runner: y = 2x + 1 (1D linear)",
                tolerance: 0.25,
                dataset: Self::build_linear_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: y = exp(0.5x) (1D exponential)",
                tolerance: 0.60,
                dataset: Self::build_exponential_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: y = x^2 (1D quadratic)",
                tolerance: 0.75,
                dataset: Self::build_quadratic_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: y = 3x1 - x2 + 0.5 (2D affine)",
                tolerance: 0.30,
                dataset: Self::build_affine_2d_dataset(n_train),
            },
        ]
    }

    fn run_case(case: DemoCase, config: &RunConfig) -> DemoSummary {
        Self::print_banner(case.title);

        let (xs, ys) = case.dataset;
        let neuron = Solution::train_one_neuron(
            xs.clone(),
            ys.clone(),
            config.lr,
            config.epochs,
        );

        println!();
        println!("{BOLD}{GREEN}✔ Training complete{RESET}");
        println!("{BOLD}Learned parameters:{RESET}");
        println!("  w = {:?}", neuron.w);
        println!("  b = {:?}", neuron.b);

        println!();
        println!("{BOLD}{YELLOW}Prediction test cases:{RESET}");

        let n = xs.len();
        let step = if config.n_show == 0 {
            1
        } else {
            (n.max(1) / config.n_show.max(1)).max(1)
        };

        let mut shown = 0usize;
        let mut i = 0usize;
        let mut passed = 0usize;
        let mut total_error = 0.0f32;

        while i < n && shown < config.n_show {
            thread::sleep(Duration::from_millis(config.delay_ms));

            let x = &xs[i];
            let y_true = ys[i];
            let y_hat = neuron.forward(x);
            let error = (y_hat - y_true).abs();
            let is_pass = error <= case.tolerance;

            if is_pass {
                passed += 1;
            }
            total_error += error;

            println!(
                "  Test {:>2}: {}{}{}  x = {:?}, y_true = {:>8.4}, y_hat = {:>8.4}, error = {:>8.4}",
                shown + 1,
                Self::status_color(is_pass),
                Self::status_label(is_pass),
                RESET,
                x,
                y_true,
                y_hat,
                error
            );

            shown += 1;
            i += step;
        }

        let score = if shown > 0 {
            100.0 * passed as f32 / shown as f32
        } else {
            0.0
        };

        let mae = if shown > 0 {
            total_error / shown as f32
        } else {
            0.0
        };

        let summary = DemoSummary {
            title: case.title.to_string(),
            shown,
            passed,
            mae,
            score,
        };

        let summary_color = Self::status_color(summary.all_passed());
        let summary_label = if summary.all_passed() {
            "ALL PASSED"
        } else {
            "SOME FAILED"
        };

        println!();
        println!(
            "{BOLD}{CYAN}Summary:{RESET} {}{}{} ({}/{})",
            summary_color,
            summary_label,
            RESET,
            summary.passed,
            summary.shown
        );
        println!("  Score:     {:>5.1}% ({})", summary.score, summary.score_label());
        println!("  MAE:       {:>5.4}", summary.mae);
        println!();

        summary
    }

    fn print_overall_summary(summaries: &[DemoSummary]) {
        let case_count = summaries.len();
        if case_count == 0 {
            return;
        }

        let mut total_score = 0.0f32;
        let mut total_mae = 0.0f32;
        let mut total_passed = 0usize;
        let mut total_shown = 0usize;

        for summary in summaries {
            total_score += summary.score;
            total_mae += summary.mae;
            total_passed += summary.passed;
            total_shown += summary.shown;
        }

        let avg_score = total_score / case_count as f32;
        let avg_mae = total_mae / case_count as f32;

        println!("{BOLD}{CYAN}Overall score:{RESET} {:>5.1}%", avg_score);
        println!("  Total passes: {}/{}", total_passed, total_shown);
        println!("  Average MAE:  {:>5.4}", avg_mae);
        println!();
    }

    fn print_runner_notes() {
        println!("{BOLD}{YELLOW}Notes:{RESET}");
        println!("  - Linear and 2D affine tasks pass because one neuron");
        println!("    computes y = w·x + b, which can represent affine rules exactly.");
        println!("  - Exponential and quadratic tasks are nonlinear, so a single");
        println!("    linear neuron can only approximate them, not fit them perfectly.");
        println!("  - That is why the quadratic section has several FAIL results:");
        println!("    the model is too simple, not necessarily incorrect.");
        println!();

        println!("{BOLD}{CYAN}Why this matters:{RESET}");
        println!("  - Step 15 gives you a full training loop end-to-end.");
        println!("  - It also shows the limit of a single linear neuron.");
        println!("  - To model curved patterns like x^2, we need nonlinear");
        println!("    activations and eventually multiple neurons/layers.");
        println!();

        println!("{BOLD}{MAGENTA}Next direction:{RESET}");
        println!("  - Keep building from this training loop.");
        println!("  - The next big idea is moving from one neuron to richer");
        println!("    models that can represent nonlinear behavior.");
        println!();
    }

    fn run(config: RunConfig) {
        Self::print_banner("NN Runner: Train One Neuron");
        println!("Each prediction below is treated like a test case.");
        println!("Green = pass, red = fail.");
        println!();

        let cases = Self::build_cases(config.n_train);
        let mut summaries = Vec::with_capacity(cases.len());

        for case in cases {
            let summary = Self::run_case(case, &config);
            summaries.push(summary);
        }

        println!("{BOLD}{CYAN}Done testing multiple datasets.{RESET}");
        println!();
        Self::print_overall_summary(&summaries);
        Self::print_runner_notes();
    }
}

fn main() {
    let config = RunConfig {
        lr: 0.05,
        epochs: 200,
        n_train: 200,
        n_show: 12,
        delay_ms: 150,
    };

    Runner::run(config);
}