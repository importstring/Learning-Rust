/*
NN Step 7. Dense Layer Forward
[src/solutions/nn/07-DenseForward.rs]

Goal:
Implement the forward pass of a single dense (fully-connected) layer:

output = ReLU(input * weights + bias)

Why this matters for neural networks:
This is the core operation in a feedforward neural network layer.
You're combining:
- matrix multiplication,
- bias addition,
- and a non-linear activation (ReLU).

Definitions and shapes:
Let:
- X be the input matrix with shape (batch_size, in_dim)
- W be the weight matrix with shape (in_dim, out_dim)
- b be the bias vector with length out_dim

Then:
1. Z = X * W has shape (batch_size, out_dim)
2. Z_b = Z + b (bias added to each row) has shape (batch_size, out_dim)
3. Y = ReLU(Z_b) has shape (batch_size, out_dim)

So dense_forward(X, W, b) should return Y.

Shape rules:
- If X has shape (B, I) and W has shape (I, O), matmul is valid.
- b must have length O.
- If shapes are invalid at any step, return None.

Examples:

Example 1:
X = [[1.0, 2.0]]         // (1 x 2)
W = [[1.0, 0.0],         // (2 x 2)
     [0.0, 1.0]]
b = [0.0, 0.0]

Z   = X * W = [[1.0, 2.0]]
Z_b = Z + b = [[1.0, 2.0]]
Y   = ReLU(Z_b) = [[1.0, 2.0]]

Example 2:
X = [[-1.0, 2.0]]        // (1 x 2)
W = [[1.0],              // (2 x 1)
     [1.0]]
b = [0.0]

Z   = [[(-1.0)*1.0 + 2.0*1.0]] = [[1.0]]
Z_b = [[1.0 + 0.0]] = [[1.0]]
Y   = [[ReLU(1.0)]] = [[1.0]]

Example 3 (negative result gets zeroed):
X = [[-1.0, -2.0]]       // (1 x 2)
W = [[1.0],              // (2 x 1)
     [1.0]]
b = [0.0]

Z   = [[(-1.0)*1.0 + (-2.0)*1.0]] = [[-3.0]]
Z_b = [[-3.0]]
Y   = [[0.0]]
*/

struct Solution;

// Assume you already have these implemented somewhere in this file:
//
// impl Solution {
//     pub fn matmul(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Option<Vec<Vec<f32>>> { ... }
//     pub fn add_bias(matrix: Vec<Vec<f32>>, bias: Vec<f32>) -> Option<Vec<Vec<f32>>> { ... }
//     pub fn relu(matrix: Vec<Vec<f32>>) -> Vec<Vec<f32>> { ... }
// }

impl Solution {

    pub fn matmul(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Option<Vec<Vec<f32>>> {
        if a.is_empty() || b.is_empty() {
            return None;
        }

        // check rectangular
        let a_cols = a[0].len();
        if a_cols == 0 {
            return None;
        }
        for row in &a {
            if row.len() != a_cols {
                return None;
            }
        }

        let b_rows = b.len();
        let b_cols = b[0].len();
        if b_cols == 0 {
            return None;
        }
        for row in &b {
            if row.len() != b_cols {
                return None;
            }
        }

        // shape rule
        if a_cols != b_rows {
            return None;
        }

        let a_rows = a.len();
        let mut result = vec![vec![0.0; b_cols]; a_rows];

        for i in 0..a_rows {
            for j in 0..b_cols {
                let mut sum = 0.0;
                for k in 0..a_cols {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }

        Some(result)
    }
    
    pub fn add_bias(matrix: Vec<Vec<f32>>, bias: Vec<f32>) -> Option<Vec<Vec<f32>>> {
        if matrix.is_empty() {
            return Some(vec![]);
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        if bias.len() != cols {
            return None;
        }

        let mut result = vec![vec![0.0; cols]; rows];

        for i in 0..rows {
            for j in 0..cols {
                result[i][j] = matrix[i][j] + bias[j];
            }
        }
        
        Some(result)
    }

    pub fn relu(matrix: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        if matrix.is_empty() {
            return vec![];
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        let mut result = vec![vec![0.0; cols]; rows];

        for (i, row) in matrix.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                result[i][j] = x.max(0.0);
            }
        }

        result
    }


    pub fn dense_forward(
        x: Vec<Vec<f32>>,
        w: Vec<Vec<f32>>,
        b: Vec<f32>,
    ) -> Option<Vec<Vec<f32>>> {
        // Steps:
        // 1. z = matmul(x, w)?        // (B x O)
        // 2. z_b = add_bias(z, b)?    // (B x O)
        // 3. y = relu(z_b)            // (B x O)
        // 4. return Some(y)

    }
}

// Tests

fn matrices_close(a: &[Vec<f32>], b: &[Vec<f32>], eps: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for i in 0..a.len() {
        if a[i].len() != b[i].len() {
            return false;
        }
        for j in 0..a[i].len() {
            if (a[i][j] - b[i][j]).abs() > eps {
                return false;
            }
        }
    }

    true
}

fn print_result(
    case_name: &str,
    x: Vec<Vec<f32>>,
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
    expected: Option<Vec<Vec<f32>>>,
) {
    let actual = Solution::dense_forward(x.clone(), w.clone(), b.clone());

    let passed = match (&actual, &expected) {
        (Some(a), Some(e)) => matrices_close(a, e, 1e-5),
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
    println!("{}X:{} {:?}", yellow, reset, x);
    println!("{}W:{} {:?}", yellow, reset, w);
    println!("{}b:{} {:?}", yellow, reset, b);
    println!("{}Expected:{} {:?}", yellow, reset, expected);
    println!("{}Actual:{}   {:?}", yellow, reset, actual);

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Dense Layer Forward\x1b[0m");

    // Identity-like layer, no bias
    print_result(
        "Test Case 1",
        vec![vec![1.0, 2.0]],
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![0.0, 0.0],
        Some(vec![vec![1.0, 2.0]]),
    );

    // 1 x 2 -> 1 x 1, positive result
    print_result(
        "Test Case 2",
        vec![vec![-1.0, 2.0]],
        vec![vec![1.0], vec![1.0]],
        vec![0.0],
        Some(vec![vec![1.0]]),
    );

    // 1 x 2 -> 1 x 1, negative result zeroed by ReLU
    print_result(
        "Test Case 3",
        vec![vec![-1.0, -2.0]],
        vec![vec![1.0], vec![1.0]],
        vec![0.0],
        Some(vec![vec![0.0]]),
    );

    // Invalid shapes: X (1x2), W (3x1)
    print_result(
        "Test Case 4",
        vec![vec![1.0, 2.0]],
        vec![vec![1.0], vec![1.0], vec![1.0]],
        vec![0.0],
        None,
    );
}