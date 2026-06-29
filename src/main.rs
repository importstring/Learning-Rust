/*
NN Step 16. Train One Dense Layer
[src/solutions/nn/16-TrainDenseLayer.rs]


Goal:
Train a single dense layer with multiple output neurons
on a tiny dataset using gradient descent.


Why this matters for neural networks:
Step 15 trained exactly one neuron:
  x -> scalar output

That is useful, but real neural networks use layers of neurons:
  x -> vector output

A dense layer is the first true “multi-neuron” building block.
It takes an input vector x of dimension D and produces
an output vector y_hat of dimension O.

This step makes you rebuild several earlier pieces together:
- matrix multiplication
- adding bias
- dense forward pass
- MSE loss over vectors
- dense backward gradients
- SGD parameter updates
- the full training loop


Setup:
We consider one dense layer with:
- weights W: Vec<Vec<f32>>   of shape (D x O)
- bias    b: Vec<f32>        of shape (O)

For one input x: &[f32] of shape (D),
the forward rule is:

  y_hat[j] = sum_i x[i] * W[i][j] + b[j]

Equivalently:

  y_hat = xW + b

There is no nonlinear activation in this step.
This is still a linear layer.


Loss:
For one sample with target y of shape (O),
use vector MSE:

  loss = 0.5 * sum_j (y_hat[j] - y[j])^2


Gradients for one sample:
Let:

  diff[j] = y_hat[j] - y[j]

Then:

  dW[i][j] = x[i] * diff[j]
  db[j]    = diff[j]

Training loop:
For each epoch:
  - initialize dW_sum and db_sum to zeros
  - for each sample (x, y):
      * compute y_hat
      * compute per-sample gradients
      * accumulate into dW_sum and db_sum
  - average gradients over N samples
  - apply one SGD update to W and b


Important idea:
This layer has multiple neurons, but it is still linear.
So it can learn multi-output affine relationships exactly,
but it still cannot perfectly model nonlinear targets
like x^2 or exp(x) by itself.


Notes:
- Follow the step comments exactly; tests are sensitive.
- Shapes matter a lot here.
- Think carefully about dimensions:
    x is length D
    W is D x O
    b is length O
    y_hat is length O
*/



struct Solution;


pub struct DenseLayer {
    pub w: Vec<Vec<f32>>, // shape: D x O
    pub b: Vec<f32>,      // shape: O
}



/// Hints:
/// 1) Reuse your earlier dense forward idea, but now think row-vector times matrix.
/// 2) Output length is O.
/// 3) Each output coordinate depends on all input coordinates.
impl DenseLayer {
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // TODO: compute y_hat = xW + b
        let d = self.w.len();
        let o = self.w[0].len();

        let mut z = vec![0.0; o];
        for i in 0..d {
            for j in 0..o {
                z[j] += x[i] * self.w[i][j];
            }
        }
        
        let mut y_hat = Vec::with_capacity(o);
        for j in 0..o {
            y_hat.push(z[j] + self.b[j]);
        }

        y_hat
    }
}

/// Hints:
/// 1) This is scalar MSE repeated across output coordinates.
/// 2) Keep the 0.5 factor.
/// 3) Be careful not to average twice unless the problem explicitly wants that.
pub fn loss_mse_vec(y_hat: &[f32], y: &[f32]) -> f32 {
    // TODO: compute vector MSE loss
    let mut out = 0.0_f32;

    for j in 0..y.len() {
        let e = y_hat[j] - y[j];
        out += 0.5 * e * e;
    }

    out
}


/// Hints:
/// 1) Let D be the input dimension and O be the output dimension.
///    D = x.len(), O = layer.b.len() (and also layer.w[0].len()).
/// 2) Compute y_hat first, then diff[j] = y_hat[j] - y[j].
/// 3) db has length O; dW has shape D x O.
///    Each dW[i][j] should match the equation dW_ij = x_i * diff_j.
pub fn dense_gradients(
    layer: &DenseLayer,
    x: &[f32],
    y: &[f32],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    // TODO: compute per-sample dW and db
    let d = layer.w.len(); // D
    let o = layer.w[0].len(); // O

    let mut db = vec![0.0; o];
    let y_hat: Vec<f32> = layer.forward(x);

    for j in 0..o {
        db[j] = y_hat[j] - y[j];
    }

    let mut dW = vec![vec![0.0; o]; d];
    for i in 0..d {
        for j in 0..o {
            dW[i][j] = x[i] * db[j];
        }
    }

    (dW, db)
}



/// Hints:
/// 1) This is Step 14 again, but over all weights and all biases in the layer.
/// 2) Parameter shapes and gradient shapes must match.
/// 3) Every parameter updates independently by subtracting lr times its gradient.
pub fn sgd_update_dense(
    layer: &mut DenseLayer,
    dW: &[Vec<f32>],
    db: &[f32],
    lr: f32,
) {
    // TODO: apply SGD update to W and b
    for i in 0..layer.w.len() {
        for j in 0..layer.w[0].len() {
            layer.w[i][j] = layer.w[i][j] - lr * dW[i][j];
        }
    }
    for j in 0..layer.w[0].len() {
        layer.b[j] = layer.b[j] - lr * db[j];
    }
}


/// Hints:
/// 1) This is Step 15 generalized from scalar output to vector output.
/// 2) Keep your attention on shapes: inputs are N x D, targets are N x O.
/// 3) The epoch structure is the same as before: accumulate, average, update.
impl Solution {
    pub fn train_dense_layer(
        xs: Vec<Vec<f32>>, // shape: N x D
        ys: Vec<Vec<f32>>, // shape: N x O
        lr: f32,
        epochs: usize,
    ) -> DenseLayer {
        // TODO: implement full dense-layer training loop
        let n = xs.len(); // Samples
        let d = xs[0].len(); // Input dimension
        let o = ys[0].len(); // Output dimension

        let mut layer = DenseLayer {
            w: vec![vec![0.0; o]; d],
            b: vec![0.0; o],
        };

        for e in 0..epochs {
            
            let mut db_sum = vec![0.0; o];
            let mut dW_sum = vec![vec![0.0; o]; d];

            for n in 0..n {
                let (dW, db) = dense_gradients(&layer, &xs[n], &ys[n]);

                for i in 0..d {
                    for j in 0..o {
                        dW_sum[i][j] += dW[i][j];
                    }
                }
                
                for j in 0..o {
                    db_sum[j] += db[j];
                }
            }

            for i in 0..d {
                for j in 0..o {
                    dW_sum[i][j] /= n as f32;
                }
            }

            for j in 0..o {
                db_sum[j] /= n as f32;
            }

            sgd_update_dense(&mut layer, &dW_sum, &db_sum, lr);
        }

        layer
    }
}

// Tests / Demo

use std::thread;
use std::time::Duration;

const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

type Dataset = (Vec<Vec<f32>>, Vec<Vec<f32>>);

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

    fn vector_max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        let mut out = 0.0_f32;
        for j in 0..a.len() {
            let err = (a[j] - b[j]).abs();
            if err > out {
                out = err;
            }
        }
        out
    }

    fn vector_mean_abs_error(a: &[f32], b: &[f32]) -> f32 {
        let mut out = 0.0_f32;
        for j in 0..a.len() {
            out += (a[j] - b[j]).abs();
        }
        out / a.len() as f32
    }

    fn build_two_output_linear_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        for i in 0..n {
            let x = -1.0 + 2.0 * (i as f32) / (n as f32);
            xs.push(vec![x]);
            ys.push(vec![2.0 * x + 1.0, -x + 0.5]);
        }

        (xs, ys)
    }

    fn build_two_input_two_output_affine_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        for i in 0..n {
            let x1 = -1.0 + 2.0 * (i as f32) / (n as f32);
            let x2 = 1.0 - 2.0 * (i as f32) / (n as f32);

            xs.push(vec![x1, x2]);
            ys.push(vec![
                3.0 * x1 - x2 + 0.5,
                x1 + 2.0 * x2 - 1.0,
            ]);
        }

        (xs, ys)
    }

    fn build_two_output_linear_shifted_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        for i in 0..n {
            let x = -2.0 + 4.0 * (i as f32) / (n as f32);
            xs.push(vec![x]);
            ys.push(vec![
                -3.0 * x + 2.0,
                0.25 * x - 1.5,
            ]);
        }

        (xs, ys)
    }

    fn build_three_input_two_output_affine_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        for i in 0..n {
            let t = -1.0 + 2.0 * (i as f32) / (n as f32);
            let x1 = t;
            let x2 = 0.5 * t + 0.25;
            let x3 = -0.75 * t + 0.1;

            xs.push(vec![x1, x2, x3]);
            ys.push(vec![
                2.0 * x1 - x2 + 0.5 * x3 + 0.25,
                -x1 + 3.0 * x2 + 2.0 * x3 - 0.75,
            ]);
        }

        (xs, ys)
    }

    fn build_quadratic_two_output_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        for i in 0..n {
            let x = -1.5 + 3.0 * (i as f32) / (n as f32);
            xs.push(vec![x]);
            ys.push(vec![
                x * x,
                0.5 * x * x - x + 1.0,
            ]);
        }

        (xs, ys)
    }

    fn build_sine_like_two_output_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        for i in 0..n {
            let x = -1.0 + 2.0 * (i as f32) / (n as f32);
            xs.push(vec![x]);
            ys.push(vec![
                x.sin(),
                (2.0 * x).sin(),
            ]);
        }

        (xs, ys)
    }

    fn build_small_range_almost_linear_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        for i in 0..n {
            let x = -0.2 + 0.4 * (i as f32) / (n as f32);
            xs.push(vec![x]);
            ys.push(vec![
                x.exp(),
                x * x + 0.5,
            ]);
        }

        (xs, ys)
    }

    fn build_cases(n_train: usize) -> Vec<DemoCase> {
        vec![
            DemoCase {
                title: "NN Runner: [2x + 1, -x + 0.5] (1D -> 2D)",
                tolerance: 0.25,
                dataset: Self::build_two_output_linear_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: affine plane -> 2 outputs (2D -> 2D)",
                tolerance: 0.30,
                dataset: Self::build_two_input_two_output_affine_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: shifted affine map (1D -> 2D)",
                tolerance: 0.25,
                dataset: Self::build_two_output_linear_shifted_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: 3-input affine map (3D -> 2D)",
                tolerance: 0.30,
                dataset: Self::build_three_input_two_output_affine_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: quadratic targets (should struggle)",
                tolerance: 0.20,
                dataset: Self::build_quadratic_two_output_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: sinusoidal targets (should struggle)",
                tolerance: 0.20,
                dataset: Self::build_sine_like_two_output_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: small-range nonlinear targets (may look decent)",
                tolerance: 0.20,
                dataset: Self::build_small_range_almost_linear_dataset(n_train),
            },
        ]
    }

    fn choose_indices_to_show(pass_flags: &[bool], n_show: usize) -> Vec<usize> {
        let n = pass_flags.len();
        if n == 0 || n_show == 0 {
            return Vec::new();
        }

        let mut chosen = Vec::new();
        let mut used = vec![false; n];

        for i in 0..n {
            if !pass_flags[i] && chosen.len() < n_show {
                chosen.push(i);
                used[i] = true;
            }
        }

        if chosen.len() < n_show {
            let remaining = n_show - chosen.len();
            let step = (n.max(1) / remaining.max(1)).max(1);

            let mut i = 0usize;
            while i < n && chosen.len() < n_show {
                if !used[i] {
                    chosen.push(i);
                    used[i] = true;
                }
                i += step;
            }
        }

        if chosen.len() < n_show {
            for i in 0..n {
                if !used[i] && chosen.len() < n_show {
                    chosen.push(i);
                    used[i] = true;
                }
            }
        }

        chosen
    }

    fn run_case(case: DemoCase, config: &RunConfig) -> DemoSummary {
        Self::print_banner(case.title);

        let (xs, ys) = case.dataset;
        let layer = Solution::train_dense_layer(
            xs.clone(),
            ys.clone(),
            config.lr,
            config.epochs,
        );

        println!();
        println!("{BOLD}{GREEN}✔ Training complete{RESET}");
        println!("{BOLD}Learned parameters:{RESET}");
        println!("  W = {:?}", layer.w);
        println!("  b = {:?}", layer.b);

        println!();
        println!("{BOLD}{YELLOW}Prediction test cases:{RESET}");

        let n = xs.len();

        let mut pass_flags = Vec::with_capacity(n);
        let mut max_errs = Vec::with_capacity(n);
        let mut mean_errs = Vec::with_capacity(n);
        let mut y_hats = Vec::with_capacity(n);

        let mut passed = 0usize;
        let mut total_error = 0.0f32;

        for i in 0..n {
            let y_hat = layer.forward(&xs[i]);
            let max_err = Self::vector_max_abs_error(&y_hat, &ys[i]);
            let mean_err = Self::vector_mean_abs_error(&y_hat, &ys[i]);
            let is_pass = max_err <= case.tolerance;

            if is_pass {
                passed += 1;
            }
            total_error += mean_err;

            pass_flags.push(is_pass);
            max_errs.push(max_err);
            mean_errs.push(mean_err);
            y_hats.push(y_hat);
        }

        let show_indices = Self::choose_indices_to_show(&pass_flags, config.n_show);

        for (shown_idx, &i) in show_indices.iter().enumerate() {
            thread::sleep(Duration::from_millis(config.delay_ms));

            let x = &xs[i];
            let y_true = &ys[i];
            let y_hat = &y_hats[i];
            let max_err = max_errs[i];
            let mean_err = mean_errs[i];
            let is_pass = pass_flags[i];

            println!(
                "  Test {:>3}: {}{}{}  x = {:?}, y_true = {:?}, y_hat = {:?}, max_err = {:>8.4}, mean_err = {:>8.4}",
                i + 1,
                Self::status_color(is_pass),
                Self::status_label(is_pass),
                RESET,
                x,
                y_true,
                y_hat,
                max_err,
                mean_err
            );

            let _ = shown_idx;
        }

        let shown = show_indices.len();

        let score = if n > 0 {
            100.0 * passed as f32 / n as f32
        } else {
            0.0
        };

        let mae = if n > 0 {
            total_error / n as f32
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

        let summary_color = if passed == n { GREEN } else { RED };
        let summary_label = if passed == n {
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
            passed,
            n
        );
        println!("  Visible:   {}/{}", shown, n);
        println!("  Score:     {:>5.1}% ({})", summary.score, summary.score_label());
        println!("  Mean MAE:  {:>5.4}", summary.mae);
        println!();

        summary
    }

    fn print_overall_summary(summaries: &[DemoSummary]) {
        let mut total_score = 0.0_f32;
        let mut total_mae = 0.0_f32;
        let mut count = 0usize;

        for s in summaries {
            total_score += s.score;
            total_mae += s.mae;
            count += 1;
        }

        let avg_score = if count > 0 {
            total_score / count as f32
        } else {
            0.0
        };

        let avg_mae = if count > 0 {
            total_mae / count as f32
        } else {
            0.0
        };

        println!("{BOLD}{CYAN}Overall summary across datasets:{RESET}");
        println!("  Avg score: {:>5.1}%", avg_score);
        println!("  Avg MAE:   {:>5.4}", avg_mae);
        println!();

        println!("{BOLD}{CYAN}Per-case recap:{RESET}");
        for s in summaries {
            println!(
                "  - {}: visible = {}, passed = {}, score = {:>5.1}%, MAE = {:>6.4}",
                s.title, s.shown, s.passed, s.score, s.mae
            );
        }
        println!();
    }

    fn print_runner_notes() {
        println!("{BOLD}{YELLOW}Notes:{RESET}");
        println!("  - This step trains multiple neurons at once in a single dense layer.");
        println!("  - Each output coordinate has its own bias and its own incoming weights.");
        println!("  - The whole layer is still linear, so it can fit affine vector mappings exactly.");
        println!();

        println!("{BOLD}{CYAN}Why this matters:{RESET}");
        println!("  - Step 15 trained one neuron.");
        println!("  - Step 16 trains a full dense layer.");
        println!("  - This is the actual unit you will later stack into deeper networks.");
        println!();

        println!("{BOLD}{MAGENTA}Next direction:{RESET}");
        println!("  - Add nonlinear activations inside training.");
        println!("  - That is what lets stacked layers represent nonlinear behavior.");
        println!();
    }

    fn run(config: RunConfig) {
        Self::print_banner("NN Runner: Train One Dense Layer");
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
        n_train: 100,
        n_show: 12,
        delay_ms: 150,
    };

    Runner::run(config);
}