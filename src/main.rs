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



// Tests (skeleton, adapt to your real test harness)

fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Train One Neuron\x1b[0m");

    // Example tiny dataset (replace with the real one from the file):
    // E.g. learn y = 2x + 1 for x in R^1.

    let xs = vec![
        vec![0.0],
        vec![1.0],
        vec![2.0],
    ];
    let ys = vec![
        1.0,
        3.0,
        5.0,
    ];

    let lr = 0.1;
    let epochs = 100;

    let neuron = Solution::train_one_neuron(xs, ys, lr, epochs);

    println!("Trained weights: {:?}", neuron.w);
    println!("Trained bias:    {:?}", neuron.b);

    // In your actual file, the tests will assert that neuron.w and neuron.b
    // are close to some expected values using eps comparisons, similar to Step 14.
}