/*
NN Step 17A. Output Delta for Dense + ReLU
[src/solutions/nn/17A-OutputDeltaDenseRelu.rs]

Goal:
Implement the output-side backprop pieces for one dense + ReLU layer,
using small sub-functions.

Why this matters for neural networks:
Before, you may have written one large gradient function directly.

Now, do it in a more structured way:
1) compute the linear output z
2) compute the activated output y_hat
3) compute diff = y_hat - y
4) compute relu'(z)
5) combine them into delta
6) use delta to build db and dW

This is closer to how backprop is actually understood:
delta is the "error signal" at each output neuron.

Layer setup:
For one input x of length D and one target y of length O:

  z[j]     = sum_i x[i] * W[i][j] + b[j]
  y_hat[j] = relu(z[j])

Loss:
We use vector MSE for one sample:

  loss = 0.5 * sum_j (y_hat[j] - y[j])^2

Backprop idea:
For each output coordinate j:

  diff[j]  = y_hat[j] - y[j]
  delta[j] = diff[j] * relu_deriv(z[j])

Then:

  db[j]    = delta[j]
  dW[i][j] = x[i] * delta[j]

Important:
- You should rely heavily on sub-functions.
- The main gradient function is already scaffolded.
- You should implement the TODO sub-functions.
- Tests are provided for each level.

What you used to do:
- Write one big function and try to reason about everything at once.

What you should do now:
- Break the math into named steps.
- Prove each step with tests.
- Then let the main function call those steps.

Shape rules:
- x has length D
- y has length O
- w has shape D x O
- b has length O
- z has length O
- y_hat has length O
- delta has length O
- db has length O
- dW has shape D x O
*/

struct Solution;

pub struct DenseReluLayer {
    pub w: Vec<Vec<f32>>, // shape: D x O
    pub b: Vec<f32>,      // shape: O
}

fn relu(z: f32) -> f32 {
    z.max(0.0)
}

impl DenseReluLayer {
    pub fn forward_linear(&self, x: &[f32]) -> Vec<f32> {
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

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let z = self.forward_linear(x);
        let mut y_hat = Vec::with_capacity(z.len());

        for j in 0..z.len() {
            y_hat.push(relu(z[j]));
        }

        y_hat
    }
}

impl Solution {
    /// Hints:
    /// 1) This is the ReLU derivative with respect to z.
    /// 2) Use 1.0 when z > 0.0.
    /// 3) Use 0.0 when z <= 0.0.
    pub fn relu_deriv(z: f32) -> f32 {
        if z > 0.0 {
            return 1.0;
        }

        0.0
    }

    /// Hints:
    /// 1) diff is "prediction minus target".
    /// 2) This is dL/dy_hat for 0.5 * (y_hat - y)^2.
    /// 3) Output is one scalar for one coordinate.
    pub fn get_diff(y_hat_j: f32, y_j: f32) -> f32 {
        y_hat_j - y_j
    }

    /// Notes:
    // there's a common term between both d_w & d_b
    // it's the first two derivitives. The change 
    // in the loss with respect to the the activiation 
    // and the change in the activiation with respect
    // to the activiation non-linear gate. So z becoming
    // a which is essentially just relu. 
    pub fn get_delta(diff_j: f32, z_j: f32) -> f32 {
        diff_j * Self::relu_deriv(z_j)
    }

    /// Hints:
    /// 1) Bias gradient for one output neuron is just delta.
    /// 2) db[j] = delta[j].
    pub fn get_db(delta_j: f32) -> f32 {
        delta_j
    }

    /// Hints:
    /// 1) Weight gradient is input times delta.
    /// 2) dW[i][j] = x[i] * delta[j].
    /// 3) One input coordinate, one output coordinate.
    pub fn get_dw(x_i: f32, delta_j: f32) -> f32 {
        x_i * delta_j
    }

   pub fn build_delta_vector(z: &[f32], y_hat: &[f32], y: &[f32]) -> Option<Vec<f32>> {
        if z.len() != y_hat.len() || y_hat.len() != y.len() {
            return None;
        }

        let mut delta = Vec::with_capacity(z.len());

        for j in 0..z.len() {
            let diff_j = Solution::get_diff(y_hat[j], y[j]);
            let delta_j = Solution::get_delta(diff_j, z[j]);
            delta.push(delta_j);
        }

        Some(delta)
    }

    pub fn build_db(delta: &[f32]) -> Vec<f32> {
        let mut d_b = Vec::with_capacity(delta.len());

        for j in 0..delta.len() {
            d_b.push(Solution::get_db(delta[j]));
        }

        d_b
    }

    pub fn build_d_w(x: &[f32], delta: &[f32]) -> Vec<Vec<f32>> {
        let d = x.len();
        let o = delta.len();

        let mut d_w = vec![vec![0.0; o]; d];

        for i in 0..d {
            for j in 0..o {
                d_w[i][j] = Solution::get_dw(x[i], delta[j]);
            }
        }

        d_w
    }

    pub fn dense_relu_gradients_from_delta(
        layer: &DenseReluLayer,
        x: &[f32],
        y: &[f32],
    ) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
        if layer.w.is_empty() {
            return None;
        }

        let d = layer.w.len();
        let o = layer.w[0].len();

        if x.len() != d || y.len() != o || layer.b.len() != o {
            return None;
        }

        for i in 0..d {
            if layer.w[i].len() != o {
                return None;
            }
        }

        let z = layer.forward_linear(x);
        let y_hat = layer.forward(x);

        let delta = Solution::build_delta_vector(&z, &y_hat, y)?;
        let d_b = Solution::build_db(&delta);
        let d_w = Solution::build_d_w(x, &delta);

        Some((d_w, d_b))
    }
}
  
/* Tests */

fn floats_close(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

fn vec_close(a: &[f32], b: &[f32], eps: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for i in 0..a.len() {
        if !floats_close(a[i], b[i], eps) {
            return false;
        }
    }

    true
}

fn matrices_close(a: &[Vec<f32>], b: &[Vec<f32>], eps: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for i in 0..a.len() {
        if a[i].len() != b[i].len() {
            return false;
        }

        for j in 0..a[i].len() {
            if !floats_close(a[i][j], b[i][j], eps) {
                return false;
            }
        }
    }

    true
}

fn print_scalar_test(case_name: &str, actual: f32, expected: f32) {
    let passed = floats_close(actual, expected, 1e-5);

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Expected:{} {}", yellow, reset, expected);
    println!("{}Actual:{}   {}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn print_vec_test(case_name: &str, actual: Option<Vec<f32>>, expected: Option<Vec<f32>>) {
    let passed = match (&actual, &expected) {
        (Some(a), Some(e)) => vec_close(a, e, 1e-5),
        (None, None) => true,
        _ => false,
    };

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn print_grad_test(
    case_name: &str,
    actual: Option<(Vec<Vec<f32>>, Vec<f32>)>,
    expected: Option<(Vec<Vec<f32>>, Vec<f32>)>,
) {
    let passed = match (&actual, &expected) {
        (Some((a_w, a_b)), Some((e_w, e_b))) => {
            matrices_close(a_w, e_w, 1e-5) && vec_close(a_b, e_b, 1e-5)
        }
        (None, None) => true,
        _ => false,
    };

    let green = "\x1b[32m";
    let red = "\x1b[31m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let bold = "\x1b[1m";
    let reset = "\x1b[0m";

    println!("\n{}{}{}{}", bold, cyan, case_name, reset);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Output Delta for Dense + ReLU\x1b[0m");

    // Scalar sub-function tests
    print_scalar_test("relu_deriv positive", Solution::relu_deriv(2.0), 1.0);
    print_scalar_test("relu_deriv zero", Solution::relu_deriv(0.0), 0.0);
    print_scalar_test("relu_deriv negative", Solution::relu_deriv(-3.0), 0.0);

    print_scalar_test("get_diff positive error", Solution::get_diff(2.5, 1.0), 1.5);
    print_scalar_test("get_diff negative error", Solution::get_diff(0.5, 1.0), -0.5);

    print_scalar_test("get_delta gate open", Solution::get_delta(3.0, 2.0), 3.0);
    print_scalar_test("get_delta gate closed", Solution::get_delta(3.0, -2.0), 0.0);

    print_scalar_test("get_db basic", Solution::get_db(-1.25), -1.25);
    print_scalar_test("get_dw basic", Solution::get_dw(2.0, -3.0), -6.0);

    // Vector delta test
    print_vec_test(
        "build_delta_vector basic",
        Solution::build_delta_vector(
            &vec![1.0, -2.0, 0.0],
            &vec![1.0, 0.0, 0.0],
            &vec![0.5, 3.0, -1.0],
        ),
        Some(vec![0.5, 0.0, 0.0]),
    );

    print_vec_test(
        "build_delta_vector shape mismatch",
        Solution::build_delta_vector(
            &vec![1.0, 2.0],
            &vec![1.0],
            &vec![0.0, 0.0],
        ),
        None,
    );

    // Full gradient test: 1D -> 2D
    let layer1 = DenseReluLayer {
        w: vec![vec![2.0, -1.0]],
        b: vec![0.5, 1.0],
    };
    let x1 = vec![3.0];
    let y1 = vec![5.0, 0.0];

    // z = [6.5, -2.0]
    // y_hat = [6.5, 0.0]
    // diff = [1.5, 0.0]
    // delta = [1.5, 0.0]
    // db = [1.5, 0.0]
    // dW = [[4.5, 0.0]]
    print_grad_test(
        "dense_relu_gradients_from_delta 1D->2D",
        Solution::dense_relu_gradients_from_delta(&layer1, &x1, &y1),
        Some((vec![vec![4.5, 0.0]], vec![1.5, 0.0])),
    );

    // Full gradient test: 2D -> 2D
    let layer2 = DenseReluLayer {
        w: vec![
            vec![1.0, -2.0],
            vec![0.5, 3.0],
        ],
        b: vec![0.0, -1.0],
    };
    let x2 = vec![2.0, -1.0];
    let y2 = vec![1.0, 4.0];

    // z[0] = 2*1 + (-1)*0.5 + 0 = 1.5
    // z[1] = 2*(-2) + (-1)*3 + (-1) = -8
    // y_hat = [1.5, 0.0]
    // diff = [0.5, -4.0]
    // delta = [0.5, 0.0]
    // db = [0.5, 0.0]
    // dW = [[1.0, 0.0], [-0.5, 0.0]]
    print_grad_test(
        "dense_relu_gradients_from_delta 2D->2D",
        Solution::dense_relu_gradients_from_delta(&layer2, &x2, &y2),
        Some((
            vec![vec![1.0, 0.0], vec![-0.5, 0.0]],
            vec![0.5, 0.0],
        )),
    );

    // Error-check test
    let bad_layer = DenseReluLayer {
        w: vec![vec![1.0, 2.0], vec![3.0]],
        b: vec![0.0, 0.0],
    };
    let x_bad = vec![1.0, 2.0];
    let y_bad = vec![0.0, 0.0];

    print_grad_test(
        "dense_relu_gradients_from_delta bad layer shape",
        Solution::dense_relu_gradients_from_delta(&bad_layer, &x_bad, &y_bad),
        None,
    );
}