/*
NN Step 3. Batched 4D Tensor MatMul
[src/solutions/nn/03-BatchedMatMul4D.rs]

Goal:
Implement batched matrix multiplication for two 4D tensors stored as
Vec<Vec<Vec<Vec<f32>>>>.

We’ll treat the last two dimensions as matrices and the first two as batch
dimensions.

Concretely:
- Let A have shape (B1, B2, M, K)
- Let B have shape (B1, B2, K, N)

For each batch index (b1, b2), you must compute:

C[b1][b2] = A[b1][b2] * B[b1][b2]

where the * is normal matrix multiplication of shape (M x K) * (K x N),
producing C[b1][b2] of shape (M x N).

The output tensor C should therefore have shape (B1, B2, M, N).

Why this matters for neural networks:
Real neural network libraries often work with batched inputs and weights,
and use batched matmul operations over higher-rank tensors. Understanding
how to generalize matrix multiplication to batched tensors is an important
step toward building more complex layers (like multi-head attention or
convolutions with im2col).

Shape rules:
Let A have shape (B1, B2, M, K) and B have shape (B1, B2, K, N).

To be valid:
- B1 must match in A and B
- B2 must match in A and B
- K (the innermost dimension of A’s matrices) must equal
  K (the third dimension of B’s tensors)
If any of these conditions fail, return None.

The resulting tensor C has shape (B1, B2, M, N).

How the computation works:
For each batch index (b1, b2), and each output cell (i, j):

C[b1][b2][i][j] = sum over k of A[b1][b2][i][k] * B[b1][b2][k][j]

So in code, that usually becomes 5 nested loops:
- over b1 (outer batch)
- over b2 (inner batch)
- over i (rows of A’s matrix)
- over j (cols of B’s matrix)
- over k (shared dimension for the dot product)

Examples:

Example 1 (1x1 batch, 2D matrices):

Input:
A shape: (1, 1, 2, 2)
A[0][0] = [[1.0, 2.0],
           [3.0, 4.0]]

B shape: (1, 1, 2, 2)
B[0][0] = [[5.0, 6.0],
           [7.0, 8.0]]

Output:
Some C shape: (1, 1, 2, 2)
C[0][0] = [[19.0, 22.0],
           [43.0, 50.0]]

Example 2 (1x2 batch):

Input:
A shape: (1, 2, 1, 2)
A[0][0] = [[1.0, 2.0]]
A[0][1] = [[3.0, 4.0]]

B shape: (1, 2, 2, 1)
B[0][0] = [[5.0],
           [6.0]]
B[0][1] = [[7.0],
           [8.0]]

Output:
Some C shape: (1, 2, 1, 1)
C[0][0] = [[17.0]]  // 1*5 + 2*6 = 17
C[0][1] = [[53.0]]  // 3*7 + 4*8 = 53

Example 3 (shape mismatch):

Input:
A shape: (1, 1, 1, 3)
B shape: (1, 1, 2, 1)

Output:
None

Explanation:
Inner dimensions 3 and 2 don’t match, so the matmul is not defined
for the last two dimensions.
*/

/* Helpful Resources:

Vec indexing:
https://doc.rust-lang.org/std/vec/struct.Vec.html

Matrix multiplication shape rule (2D case):
https://en.wikipedia.org/wiki/Matrix_multiplication

Matrix multiplication algorithm (2D case, generalizes to batched):
https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
*/

struct Solution;

impl Solution {
    /// Batched 4D tensor matmul.
    ///
    /// A: shape (B1, B2, M, K)
    /// B: shape (B1, B2, K, N)
    /// Returns: Some(C) with shape (B1, B2, M, N) on success, or None on invalid shapes.
    pub fn batched_matmul4d(
        a: Vec<Vec<Vec<Vec<f32>>>>,
        b: Vec<Vec<Vec<Vec<f32>>>>,
    ) -> Option<Vec<Vec<Vec<Vec<f32>>>>> {
    }
}

// Tests

fn tensors_close(
    a: &[Vec<Vec<Vec<f32>>>>,
    b: &[Vec<Vec<Vec<f32>>>>,
    eps: f32,
) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for b1 in 0..a.len() {
        if a[b1].len() != b[b1].len() {
            return false;
        }
        for b2 in 0..a[b1].len() {
            if a[b1][b2].len() != b[b1][b2].len() {
                return false;
            }
            for i in 0..a[b1][b2].len() {
                if a[b1][b2][i].len() != b[b1][b2][i].len() {
                    return false;
                }
                for j in 0..a[b1][b2][i].len() {
                    if (a[b1][b2][i][j] - b[b1][b2][i][j]).abs() > eps {
                        return false;
                    }
                }
            }
        }
    }

    true
}

fn print_result(
    case_name: &str,
    a: Vec<Vec<Vec<Vec<f32>>>>,
    b: Vec<Vec<Vec<Vec<f32>>>>,
    expected: Option<Vec<Vec<Vec<Vec<f32>>>>>,
) {
    let actual = Solution::batched_matmul4d(a.clone(), b.clone());

    let passed = match (&actual, &expected) {
        (Some(actual_tensor), Some(expected_tensor)) => {
            tensors_close(actual_tensor, expected_tensor, 1e-5)
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
    println!("{}Expected shape:{} {:?}", yellow, reset, expected.as_ref().map(|t| (
        t.len(),
        t[0].len(),
        t[0][0].len(),
        t[0][0][0].len()
    )));
    println!("{}Actual shape:{}   {:?}", yellow, reset, actual.as_ref().map(|t| (
        t.len(),
        t[0].len(),
        t[0][0].len(),
        t[0][0][0].len()
    )));

    if passed {
        println!("{}✅ PASS{}", green, reset);
    } else {
        println!("{}❌ FAIL{}", red, reset);
    }
}

fn main() {
    println!("\x1b[1m\x1b[35mNN Runner: Batched 4D MatMul\x1b[0m");

    // Example 1: 1x1 batch, 2x2 matrices
    let a1 = vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]];
    let b1 = vec![vec![vec![vec![5.0, 6.0], vec![7.0, 8.0]]]];
    let c1 = vec![vec![vec![vec![19.0, 22.0], vec![43.0, 50.0]]]];

    print_result("Test Case 1", a1, b1, Some(c1));

    // Example 2: 1x2 batch, 1x2 * 2x1
    let a2 = vec![
        vec![
            vec![vec![1.0, 2.0]],   // (1 x 2)
            vec![vec![3.0, 4.0]],   // (1 x 2)
        ]
    ];
    let b2 = vec![
        vec![
            vec![vec![5.0], vec![6.0]],   // (2 x 1)
            vec![vec![7.0], vec![8.0]],   // (2 x 1)
        ]
    ];
    let c2 = vec![
        vec![
            vec![vec![17.0]],  // 1*5 + 2*6 = 17
            vec![vec![53.0]],  // 3*7 + 4*8 = 53
        ]
    ];

    print_result("Test Case 2", a2, b2, Some(c2));

    // Example 3: invalid inner shapes
    let a3 = vec![vec![vec![vec![1.0, 2.0, 3.0]]]]; // (1,1,1,3)
    let b3 = vec![vec![vec![vec![4.0], vec![5.0]]]]; // (1,1,2,1)

    print_result("Test Case 3", a3, b3, None);
}