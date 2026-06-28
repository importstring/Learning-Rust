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
/// 1) You already did matrix multiplication earlier; this is the same idea,
///    except x is a single row vector and W is D x O.
/// 2) Output length is O = W[0].len().
/// 3) For each output column j, sum x[i] * W[i][j] over i, then add b[j].
impl DenseLayer {
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // TODO: compute y_hat = xW + b
        //
        // HARD HINT 1:
        // If x has length D and W has shape D x O,
        // then y_hat must have length O.
        //
        // HARD HINT 2:
        // This is NOT W*x in the column-vector sense used in some textbooks.
        // Use the convention already implied by your earlier dense forward work:
        // output[j] depends on all input coordinates i.
        //
        // HARD HINT 3:
        // Skeleton:
        // let out_dim = self.b.len();
        // let mut out = vec![0.0; out_dim];
        // for j in 0..out_dim {
        //     let mut z = 0.0;
        //     for i in 0..x.len() {
        //         z += x[i] * self.w[i][j];
        //     }
        //     out[j] = z + self.b[j];
        // }
        // out
    }
}



/// Hints:
/// 1) This is scalar MSE extended over a vector output.
/// 2) Sum over output coordinates j.
/// 3) Keep the 0.5 so the derivative stays clean.
pub fn loss_mse_vec(y_hat: &[f32], y: &[f32]) -> f32 {
    // TODO: compute vector MSE loss
    //
    // HARD HINT 1:
    // You are summing scalar losses:
    // 0.5 * (y_hat[j] - y[j])^2
    //
    // HARD HINT 2:
    // Do NOT divide by output dimension unless the tests/comments require it.
    // This step is probably using "sum over outputs" for one sample,
    // then averaging across samples in the training loop.
    //
    // HARD HINT 3:
    // Start with:
    // let mut loss = 0.0;
    // for j in 0..y.len() {
    //     let diff = y_hat[j] - y[j];
    //     loss += 0.5 * diff * diff;
    // }
    // loss
}



/// Hints:
/// 1) For one sample, compute dW and db for the entire dense layer.
/// 2) Reuse the fact that dL/dy_hat[j] = y_hat[j] - y[j].
/// 3) Each weight W[i][j] connects x[i] to output j.
pub fn dense_gradients(
    layer: &DenseLayer,
    x: &[f32],
    y: &[f32],
) -> (Vec<Vec<f32>>, Vec<f32>) {
    // TODO: compute per-sample dW and db
    //
    // HARD HINT 1:
    // First get y_hat by calling forward.
    // Then diff[j] = y_hat[j] - y[j].
    //
    // HARD HINT 2:
    // db is just diff.
    //
    // HARD HINT 3:
    // dW has the same shape as W.
    // For every input coordinate i and output coordinate j:
    // dW[i][j] = x[i] * diff[j]
    //
    // HARD HINT 4:
    // This is the outer product:
    // x (length D)  outer  diff (length O)
    // producing a D x O matrix.
}



/// Hints:
/// 1) This is Step 14 again, but now for a full dense layer.
/// 2) W and dW have the same shape, b and db have the same length.
/// 3) Update every parameter independently.
pub fn sgd_update_dense(
    layer: &mut DenseLayer,
    dW: &[Vec<f32>],
    db: &[f32],
    lr: f32,
) {
    // TODO: apply SGD update to W and b
    //
    // HARD HINT 1:
    // For every i, j:
    // layer.w[i][j] -= lr * dW[i][j]
    //
    // HARD HINT 2:
    // For every j:
    // layer.b[j] -= lr * db[j]
    //
    // HARD HINT 3:
    // If your earlier Step 14 took ownership and returned a new Params,
    // this one is the in-place version for convenience.
}



/// Hints:
/// 1) This is Step 15 generalized from one output to many outputs.
/// 2) Initialize W as zeros of shape D x O and b as zeros of shape O.
/// 3) Accumulate matrix gradients dW_sum and vector gradients db_sum
///    over the whole dataset, then average, then update.
impl Solution {
    pub fn train_dense_layer(
        xs: Vec<Vec<f32>>, // shape: N x D
        ys: Vec<Vec<f32>>, // shape: N x O
        lr: f32,
        epochs: usize,
    ) -> DenseLayer {
        // TODO: implement full dense-layer training loop
        //
        // HARD HINT 1:
        // Input dimension:
        // let in_dim = xs[0].len();
        //
        // Output dimension:
        // let out_dim = ys[0].len();
        //
        // HARD HINT 2:
        // Initialize:
        // w = vec![vec![0.0; out_dim]; in_dim]
        // b = vec![0.0; out_dim]
        //
        // HARD HINT 3:
        // For each epoch:
        //   dW_sum shape = in_dim x out_dim
        //   db_sum shape = out_dim
        //
        // HARD HINT 4:
        // For each sample i:
        //   x = &xs[i]
        //   y = &ys[i]
        //   let (dW_i, db_i) = dense_gradients(&layer, x, y);
        //   accumulate into dW_sum and db_sum
        //
        // HARD HINT 5:
        // Average by N before updating.
        // Exactly like Step 15, but now:
        //   dW_avg[i][j] = dW_sum[i][j] / N
        //   db_avg[j]    = db_sum[j] / N
        //
        // HARD HINT 6:
        // Then call sgd_update_dense(&mut layer, &dW_avg, &db_avg, lr)
        //
        // HARD HINT 7:
        // If you get shape confusion, write these on paper:
        // x:      D
        // y_hat:  O
        // diff:   O
        // dW:     D x O
        // db:     O
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
        // TODO: return max_j |a[j] - b[j]|
        //
        // HARD HINT:
        // Track the maximum absolute coordinate error across the output vector.
        // Start from 0.0 and update with max().
    }


    fn vector_mean_abs_error(a: &[f32], b: &[f32]) -> f32 {
        // TODO: return mean_j |a[j] - b[j]|
        //
        // HARD HINT:
        // Sum absolute coordinate errors, then divide by vector length.
    }


    fn build_two_output_linear_dataset(n: usize) -> Dataset {
        // TODO:
        // x has shape [1]
        // y = [2x + 1, -x + 0.5]
        //
        // HARD HINT:
        // This is the cleanest "multi-neuron but still linear" test.
    }


    fn build_two_input_two_output_affine_dataset(n: usize) -> Dataset {
        // TODO:
        // x = [x1, x2]
        // y = [
        //   3x1 - x2 + 0.5,
        //   x1 + 2x2 - 1.0,
        // ]
        //
        // HARD HINT:
        // This is the true dense-layer demo:
        // multiple inputs, multiple outputs, all affine.
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
        ]
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
            let y_true = &ys[i];
            let y_hat = layer.forward(x);


            let max_err = Self::vector_max_abs_error(&y_hat, y_true);
            let mean_err = Self::vector_mean_abs_error(&y_hat, y_true);
            let is_pass = max_err <= case.tolerance;


            if is_pass {
                passed += 1;
            }
            total_error += mean_err;


            println!(
                "  Test {:>2}: {}{}{}  x = {:?}, y_true = {:?}, y_hat = {:?}, max_err = {:>8.4}, mean_err = {:>8.4}",
                shown + 1,
                Self::status_color(is_pass),
                Self::status_label(is_pass),
                RESET,
                x,
                y_true,
                y_hat,
                max_err,
                mean_err
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
        println!("  Mean MAE:  {:>5.4}", summary.mae);
        println!();


        summary
    }


    fn print_overall_summary(summaries: &[DemoSummary]) {
        // TODO: aggregate overall score/MAE across cases
        //
        // HARD HINT:
        // This is the same pattern you already wrote in Step 15.
        // Sum scores, MAEs, passes, and shown counts across summaries.
        // Then average score and MAE over number of cases.
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
        n_train: 200,
        n_show: 12,
        delay_ms: 150,
    };


    Runner::run(config);
}