/*
NN Step 17. Train One Nonlinear Dense Layer
[src/solutions/nn/17-NonlinearDenseLayer.rs]

Goal:
Train a single dense layer with multiple output neurons,
but now with a nonlinear activation applied to the outputs.

Why this matters for neural networks:
Step 16 trained a purely linear dense layer:
  x -> xW + b

That layer can only represent affine mappings and will
struggle on nonlinear targets like x^2 or sin(x).

Real neural networks insert nonlinear activations between
dense layers, for example:
  x -> ReLU(xW + b)

This step introduces that idea in the simplest way:
a single dense layer followed by ReLU on each output.

ReLU (Rectified Linear Unit):
  relu(z) = max(0, z)

We apply it coordinate-wise to the pre-activation vector z.

This makes the layer nonlinear and lets it approximate
some nonlinear behaviors, especially piecewise-linear ones.

Setup:
We consider one dense + ReLU layer with:
- weights W: Vec<Vec<f32>>   of shape (D x O)
- bias    b: Vec<f32>        of shape (O)

For one input x: &[f32] of shape (D),
the pre-activation z and activated output y_hat are:

  z[j]     = sum_i x[i] * W[i][j] + b[j]
  y_hat[j] = relu(z[j])

Equivalently:

  z     = xW + b
  y_hat = relu(z)

Loss:
For one sample with target y of shape (O),
we still use vector MSE:

  loss = 0.5 * sum_j (y_hat[j] - y[j])^2

Gradients for one sample:
Let:

  diff[j]  = y_hat[j] - y[j]
  relu'(z) = 1 if z > 0, else 0

Then the effective output error after ReLU is:

  delta[j] = diff[j] * relu'(z[j])

and the parameter gradients become:

  dW[i][j] = x[i] * delta[j]
  db[j]    = delta[j]

Training loop:
For each epoch:
  - initialize dW_sum and db_sum to zeros
  - for each sample (x, y):
      * compute z and y_hat
      * compute per-sample gradients via delta
      * accumulate into dW_sum and db_sum
  - average gradients over N samples
  - apply one SGD update to W and b

Important idea:
Adding ReLU makes even a single layer nonlinear.
It still cannot perfectly represent arbitrary functions,
but it can approximate more complex, piecewise-linear
relationships than a purely affine layer.

Notes:
- Follow the step comments exactly; tests are sensitive.
- Shapes still matter a lot:
    x is length D
    W is D x O
    b is length O
    z is length O
    y_hat is length O
- Keep the 0.5 factor in the MSE loss.
*/

struct Solution;

pub struct DenseReluLayer {
    pub w: Vec<Vec<f32>>, // shape: D x O
    pub b: Vec<f32>,      // shape: O
}

/// Hints:
/// 1) ReLU is applied coordinate-wise.
/// 2) The rule is max(0, z).
/// 3) Keep it scalar here; vector application happens elsewhere.
fn relu(z: f32) -> f32 {
    // TODO: return relu(z)
    z.max(0.0)
}

/// Hints:
/// 1) This is the derivative with respect to the pre-activation z.
/// 2) Use 1 when z > 0, otherwise 0.
/// 3) At z = 0, use 0 for this project.
fn relu_deriv(z: f32) -> f32 {
    if z > 0 {
        return 1.0;
    }

    0.0
}

/// Hints:
/// 1) First compute the linear pre-activation z = xW + b.
/// 2) This is exactly the dense forward from Step 16, but returned as z.
/// 3) Output length is O.
impl DenseReluLayer {
    pub fn forward_linear(&self, x: &[f32]) -> Vec<f32> {
        // TODO: compute z = xW + b
        let d = self.w.len();
        let o = self.w[0].len();
        
        let mut z = vec![0.0; o];

        for i in 0..d {
            for j in 0..o {
                z[j] += x[i] * self.w[i][j];
            }
        }

        for j in 0..o {
            z[j] += self.b[j];
        }

        z
    }

    /// Hints:
    /// 1) Reuse forward_linear.
    /// 2) Then apply ReLU to each coordinate of z.
    /// 3) Output length is O.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // TODO: compute y_hat = relu(forward_linear(x))
        let mut a = Vec::with_capacity(x.len());
 
        for z in self.forward_linear(x) {
            a.push(relu(z));
        }

        a
    }
}

/// Hints:
/// 1) Same vector MSE as Step 16.
/// 2) Keep the 0.5 factor.
/// 3) Do not average inside this function.
pub fn loss_mse_vec(y_hat: &[f32], y: &[f32]) -> f32 {
    let mut loss = 0.0;

    for i in 0..y.len() {
        let e = y_hat[i] - y[i];
        loss += 0.5 * e * e;
    }

    loss
}

/// Hints:
/// 1) Compute z first, then y_hat.
/// 2) diff[j] = y_hat[j] - y[j].
/// 3) delta[j] = diff[j] * relu_deriv(z[j]).
/// 4) db has length O; dW has shape D x O.
/// 5) Each dW[i][j] = x[i] * delta[j].
pub fn dense_relu_gradients(
    layer: &DenseReluLayer,
    x: &[f32],
    y: &[f32],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    // TODO: compute per-sample dW and db for dense + ReLU
    let d = layer.w.len(); // Input dimension
    let o = layer.w[0].len(); // Output dimension

    let mut d_b = vec![0.0; o]; // Derivitive with respect to the bias (b)
    let mut d_w = vec![vec![0.0; o]; d]; // Derivitive with respect to the wieght (W)

    let y_hat = layer.forward(x);
    let z = layer.forward_linear(x);

    for i in 0..d {
        for j in 0..o {
            d_w[i][j] = (y_hat[j]-y[j]) * (relu_deriv(z[j])) * (x[j]);
        }
    }

    for j in 0..o {
        d_b[j] = (y_hat[j]-y[j]) * (relu_deriv(z[j]));
    }

    (d_w, d_b)
}

/// Hints:
/// 1) Same SGD update pattern as Step 16.
/// 2) Parameter shapes and gradient shapes must match.
/// 3) Each parameter subtracts lr * gradient.
pub fn sgd_update_dense_relu(
    layer: &mut DenseReluLayer,
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
/// 1) Same epoch structure as Step 16.
/// 2) Accumulate dW/db over all samples, average, then update once.
/// 3) Inputs are N x D, targets are N x O.
impl Solution {
    pub fn train_dense_relu_layer(
        xs: Vec<Vec<f32>>, // shape: N x D
        ys: Vec<Vec<f32>>, // shape: N x O
        lr: f32,
        epochs: usize,
    ) -> DenseReluLayer {
        // TODO: implement full training loop for dense + ReLU
        unimplemented!()
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

    fn build_piecewise_linear_dataset(n: usize) -> Dataset {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);

        for i in 0..n {
            let x = -1.0 + 2.0 * (i as f32) / (n as f32);
            xs.push(vec![x]);

            let y1 = (2.0 * x - 0.5).max(0.0);
            let y2 = (-x + 0.25).max(0.0);
            ys.push(vec![y1, y2]);
        }

        (xs, ys)
    }

    fn build_cases(n_train: usize) -> Vec<DemoCase> {
        vec![
            DemoCase {
                title: "NN Runner: piecewise-linear ReLU targets (1D -> 2D)",
                tolerance: 0.20,
                dataset: Self::build_piecewise_linear_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: quadratic targets (nonlinear)",
                tolerance: 0.20,
                dataset: Self::build_quadratic_two_output_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: sinusoidal targets (nonlinear)",
                tolerance: 0.20,
                dataset: Self::build_sine_like_two_output_dataset(n_train),
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
        let layer = Solution::train_dense_relu_layer(
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

        for &i in show_indices.iter() {
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
    }

    fn print_runner_notes() {
        println!("{BOLD}{YELLOW}Notes:{RESET}");
        println!("  - This step adds a ReLU activation to a dense layer.");
        println!("  - The layer is now nonlinear and can represent piecewise-linear behavior.");
        println!("  - It still has limited capacity but is more expressive than a purely affine layer.");
        println!();

        println!("{BOLD}{CYAN}Why this matters:{RESET}");
        println!("  - Step 16 trained a linear dense layer.");
        println!("  - Step 17 trains a dense layer with ReLU activation.");
        println!("  - This is the pattern you will later use between multiple stacked layers.");
        println!();

        println!("{BOLD}{MAGENTA}Next direction:{RESET}");
        println!("  - Stack multiple dense+ReLU layers.");
        println!("  - That is what builds deep nonlinear neural networks.");
        println!();
    }

    fn run(config: RunConfig) {
        Self::print_banner("NN Runner: Train One Nonlinear Dense Layer");
        println!("Each prediction below is treated like a test case.");
        println!("Green = pass, red = fail.");
        println!();

        let cases = Self::build_cases(config.n_train);
        let mut summaries = Vec::with_capacity(cases.len());

        for case in cases {
            let summary = Self::run_case(case, &config);
            summaries.push(summary);
        }

        println!("{BOLD}{CYAN}Done testing multiple nonlinear datasets.{RESET}");
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