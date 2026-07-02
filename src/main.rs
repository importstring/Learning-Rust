/*
NN Step 18. Train Two Layers: Hidden ReLU + Linear Output
[src/solutions/nn/18-HiddenReluLinearOutput.rs]


Goal:
Train a two-layer neural network with:
- one hidden dense layer followed by ReLU
- one output dense layer with no activation

This keeps the hidden layer nonlinear while making the output
layer purely linear.


Why this matters for neural networks:
Step 17 trained a single dense + ReLU layer:
  x -> ReLU(xW + b)

That already introduced nonlinearity, but it still has only one
trainable affine transform before the activation.

A more standard next step is to separate the network into:
  x -> hidden linear -> hidden ReLU -> output linear

That gives us:
  x -> ReLU(xW1 + b1)W2 + b2

Now the model can learn nonlinear hidden features first,
then combine those features linearly at the output.

This is the first real multi-layer network in your series.
It is also the first step where hidden-layer backprop matters.


Network structure:
We use two layers:

1) Hidden layer
   z1[h] = sum_i x[i] * W1[i][h] + b1[h]
   a1[h] = relu(z1[h])

2) Output layer
   z2[j] = sum_h a1[h] * W2[h][j] + b2[j]
   y_hat[j] = z2[j]

So the output layer is linear.


Why use a linear output here:
- It keeps the hidden layer nonlinear.
- It keeps the output layer simpler to differentiate.
- It is a very clean setup for regression with MSE.


Loss:
For one sample with target y of shape (O):

  loss = 0.5 * sum_j (y_hat[j] - y[j])^2


Backprop idea:
Output layer:
Since y_hat = z2, the output delta is just:

  delta2[j] = y_hat[j] - y[j]

Then:

  dW2[h][j] = a1[h] * delta2[j]
  db2[j]    = delta2[j]

Hidden layer:
The hidden neurons affect the loss through the output layer.
So for each hidden coordinate h:

  back_h = sum_j W2[h][j] * delta2[j]
  delta1[h] = back_h * relu'(z1[h])

Then:

  dW1[i][h] = x[i] * delta1[h]
  db1[h]    = delta1[h]


Training loop:
For each epoch:
  - initialize dW1_sum, db1_sum, dW2_sum, db2_sum to zeros
  - for each sample (x, y):
      * forward pass: z1, a1, y_hat
      * compute output delta2
      * compute hidden delta1
      * compute per-sample gradients
      * accumulate all gradients
  - average gradients over N samples
  - apply one SGD update to both layers


Important idea:
This is the first step where you backpropagate error from one
layer into an earlier hidden layer.

The output layer tells you how wrong the prediction is.
The hidden layer receives that signal through W2, then gets gated
by relu'(z1).

That is the essence of multi-layer backpropagation.


Notes:
- Follow the step comments exactly; tests are sensitive.
- Shapes matter a lot:
    x is length D
    W1 is D x H
    b1 is length H
    z1 is length H
    a1 is length H
    W2 is H x O
    b2 is length O
    y_hat is length O
- Keep the 0.5 factor in the MSE loss.
- Use small nonzero initialization, not all zeros.
*/


struct Solution;


pub struct TwoLayerNet {
    pub w1: Vec<Vec<f32>>, // shape: D x H
    pub b1: Vec<f32>,      // shape: H
    pub w2: Vec<Vec<f32>>, // shape: H x O
    pub b2: Vec<f32>,      // shape: O
}


/// Hints:
/// 1) ReLU is applied coordinate-wise.
/// 2) The rule is max(0, z).
/// 3) Keep it scalar here; vector application happens elsewhere.
fn relu(z: f32) -> f32 {
    z.max(0.0)
}


/// Hints:
/// 1) This is the derivative with respect to the pre-activation z.
/// 2) Use 1 when z > 0, otherwise 0.
/// 3) At z = 0, use 0 for this project.
fn relu_deriv(z: f32) -> f32 {
    if z > 0.0 {
        return 1.0;
    }

    0.0
}


/// Hints:
/// 1) First compute the hidden pre-activation z1 = xW1 + b1.
/// 2) Then compute hidden activation a1 = relu(z1).
/// 3) Then compute output y_hat = a1W2 + b2.
/// 4) The output layer is linear, so no activation there.
impl TwoLayerNet {
    pub fn forward_hidden_linear(&self, x: &[f32]) -> Vec<f32> {
        let mut z = self.b1.clone();
        
        for i in 0..self.w1.len() {
            for j in 0..self.w1[0].len() {
                z[j] += x[i] * self.w1[i][j];
            }
        }

        z
    }

    pub fn forward_hidden_activation(&self, x: &[f32]) -> Vec<f32> {
        let z1 = self.forward_hidden_linear(x);
        
        let mut a1 = Vec::with_capacity(z1.len());

        for z in z1 {
            a1.push(relu(z));
        }

        a1
    }

    pub fn forward_output(&self, a1: &[f32]) -> Vec<f32> {
        let mut z = self.b2.clone();

        for i in 0..self.w2.len() {
            for j in 0..self.w2[0].len() {
                z[j] += x[i] * self.w2[i][j];
            }
        }

        z
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let a1 = self.forward_hidden_activation(x);
        self.forward_output(&a1)
    }
}


/// Hints:
/// 1) Same vector MSE as earlier steps.
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
/// 1) For a linear output layer with MSE, delta2[j] = y_hat[j] - y[j].
/// 2) This is the same diff term as before, but no ReLU gate at output.
pub fn output_delta_linear(y_hat: &[f32], y: &[f32]) -> Vec<f32> {
    unimplemented!()
}


/// Hints:
/// 1) back_h = sum_j W2[h][j] * delta2[j].
/// 2) Then gate by relu_deriv(z1[h]).
/// 3) Output length is H.
pub fn hidden_delta_relu(
    net: &TwoLayerNet,
    z1: &[f32],
    delta2: &[f32],
) -> Vec<f32> {
    unimplemented!()
}


/// Hints:
/// 1) dW2[h][j] = a1[h] * delta2[j].
/// 2) db2[j] = delta2[j].
pub fn output_layer_gradients(
    a1: &[f32],
    delta2: &[f32],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    unimplemented!()
}


/// Hints:
/// 1) dW1[i][h] = x[i] * delta1[h].
/// 2) db1[h] = delta1[h].
pub fn hidden_layer_gradients(
    x: &[f32],
    delta1: &[f32],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    unimplemented!()
}


/// Hints:
/// 1) Forward pass first: z1, a1, y_hat.
/// 2) Then compute delta2.
/// 3) Then backprop into delta1.
/// 4) Then build gradients for both layers.
pub fn two_layer_gradients(
    net: &TwoLayerNet,
    x: &[f32],
    y: &[f32],
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
    unimplemented!()
}


/// Hints:
/// 1) Same SGD pattern as before, but now update both layers.
/// 2) Shapes must match their gradient tensors exactly.
pub fn sgd_update_two_layer(
    net: &mut TwoLayerNet,
    d_w1: &[Vec<f32>],
    d_b1: &[f32],
    d_w2: &[Vec<f32>],
    d_b2: &[f32],
    lr: f32,
) {
    unimplemented!()
}


/// Hints:
/// 1) Same epoch structure as earlier steps.
/// 2) Accumulate all four gradient blocks over all samples.
/// 3) Average, then update once per epoch.
impl Solution {
    pub fn train_two_layer_relu_linear(
        xs: Vec<Vec<f32>>, // shape: N x D
        ys: Vec<Vec<f32>>, // shape: N x O
        hidden_dim: usize,
        lr: f32,
        epochs: usize,
    ) -> TwoLayerNet {
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
    hidden_dim: usize,
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
                title: "NN Runner: piecewise-linear targets (should fit well)",
                tolerance: 0.20,
                dataset: Self::build_piecewise_linear_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: quadratic targets (should improve vs single layer)",
                tolerance: 0.20,
                dataset: Self::build_quadratic_two_output_dataset(n_train),
            },
            DemoCase {
                title: "NN Runner: sinusoidal targets (still challenging)",
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
        let net = Solution::train_two_layer_relu_linear(
            xs.clone(),
            ys.clone(),
            config.hidden_dim,
            config.lr,
            config.epochs,
        );

        println!();
        println!("{BOLD}{GREEN}✔ Training complete{RESET}");
        println!("{BOLD}Learned parameters:{RESET}");
        println!("  W1 = {:?}", net.w1);
        println!("  b1 = {:?}", net.b1);
        println!("  W2 = {:?}", net.w2);
        println!("  b2 = {:?}", net.b2);

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
            let y_hat = net.forward(&xs[i]);
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
        println!("  - This step introduces a hidden ReLU layer and a linear output layer.");
        println!("  - The output error is now backpropagated into hidden neurons.");
        println!("  - This is the first real multi-layer backprop step in the series.");
        println!();

        println!("{BOLD}{CYAN}Why this matters:{RESET}");
        println!("  - Step 17 trained one nonlinear dense layer.");
        println!("  - Step 18 trains a two-layer network: hidden ReLU + linear output.");
        println!("  - This is the standard bridge into deeper neural networks.");
        println!();

        println!("{BOLD}{MAGENTA}Next direction:{RESET}");
        println!("  - Add another hidden layer or swap in classification losses.");
        println!("  - Add gradient checking to verify your backprop numerically.");
        println!();
    }

    fn run(config: RunConfig) {
        Self::print_banner("NN Runner: Train Two Layers (Hidden ReLU + Linear Output)");
        println!("Each prediction below is treated like a test case.");
        println!("Green = pass, red = fail.");
        println!();

        let cases = Self::build_cases(config.n_train);
        let mut summaries = Vec::with_capacity(cases.len());

        for case in cases {
            let summary = Self::run_case(case, &config);
            summaries.push(summary);
        }

        println!("{BOLD}{CYAN}Done testing multiple datasets with a two-layer network.{RESET}");
        println!();
        Self::print_overall_summary(&summaries);
        Self::print_runner_notes();
    }
}


fn main() {
    let config = RunConfig {
        hidden_dim: 4,
        lr: 0.02,
        epochs: 400,
        n_train: 100,
        n_show: 12,
        delay_ms: 150,
    };

    Runner::run(config);
}
