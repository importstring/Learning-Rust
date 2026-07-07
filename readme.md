## Learning AI without Python Libraries

So I've been doing ML projects for a while now but I've realized I've just been calling libraries and calling it machine learning. Sure it works but I'm not understanding deeply what's going on. That's why I'm learning Rust. It will force me to learn algorithms, in-depth how AI works, and fill my curiousity for the language.

My final challenge will be building a neural network and I'll be working up to it by building mini projects and solving easy–hard leatcode problems.

The plan is to turn this into this very helpful environment with hints and syntax suggestions and links to Kaggle and Leatcode to make learning Rust more fun.

I'm working in `src/solutions/nn` right now!

Follow along and star!
--> can find the dev journal in root

### 📌 Today's Pinned Entries

<h4 style="margin-bottom:0;">First Weeks into Learning Rust</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-22 · Rust Learning week 3</em>
</p>

So I started on this journey realizing I had no idea what the functions I was calling were doing. I'd apply a softmax to some values that seemed very random to me following documentation, writing in my notes what intuitively everything did but I truely didn't understand what I was writing.

I've been working on MSE, matrix multiplication, dot products, relu, and then working that together to create a super simply feed forward network.

Basically I'm building myself up from the basics so that I have a strong understanding of how everything works.

It worked super nicely because I'd build in the bias function, the ReLu, and dot products and then on the very next mini challenge, use my old code for the new problem.

```Rust
pub fn matmul(...) { ...}
pub fn add_bias(...) {...}
pub fn relu(...) {...}

pub fn dense_forward(
    x: Vec<Vec<f32>>,
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
) -> Option<Vec<Vec<f32>>> {
    if x.is_empty() || w.is_empty() || b.is_empty() {
        return None;
    }

    let z = Self::matmul(x, w);
    let z_b = Self::add_bias(z?, b);
    let y = Self::relu(z_b?);
    Some(y)
```

How I've been learning thusfar:
I've been leaning on AI to roadmap my progression from skillset to skillset and have it generate LeatCode style files for me to build ML stuff. It's been working great but I'm hoping I can reach out to someone more senior for some guidance on how they'd go from here.

```Rust
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

  // ...
}
```

<h4 style="margin-bottom:0;">Moving into Softmax</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-23 · Rust Learning week 4</em>
</p>

Softmax has kinda just been this mysterious function I'd call to turn this seemingly abitrary numbers into probablilities. Now I understand how it funcitons. It's really clean and also cool that it uses `e` a lot because that means the derative will be interesting to learn about later on down the road for maybe backprop.

So for softmax, the problem I did was a row by row softmax calculation. I wrote a general function, and then this sub function:

```Rust
pub fn softmax_row(xs: &Vec<f32>) -> Vec<f32> {
    let mut denominator = 0.0_f32;
    let mut out = Vec::new();

    for j in xs {
        denominator += j.exp();
    }

    for i in xs {
        out.push( i.exp() / denominator);
    }

    out
}
```

I decided to calculate the denominator first because I realized that exponential equations can take up a lot of memory and down the road that will be a concern once the numbers are larger. Basically I pulled up Victor Zhou's amazing article and implemented in Rust. Super cool dude! Basically copied this equation.

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

Other learnings:

```Rust
Vec::new(); // I'm prefering this over vec![]; currently
for row in &matrix { ... } // because row inherits the val from matrix
// it must be borrowed instead of just for row in matrix
let x = 43.32_f32; // Learned that I can just do _f32 and it's pretty

```

<h4 style="margin-bottom:0;">Built a multi-neuron, 2d net</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-27 · Week 5</em>
</p>

Stil yet to add randomness to the wieghts and bias tuning process. Today I discovered this was really really hurting performance. Basically every neuron had the exact same wieght and bias. Basically a symetry, The ideal would be for each to symbolize a different trend and pattern in the data and then together they'll fidn the pattern.

I'm planning on doing a new activity where I do simply a bunch of exercises of just initializing a bunch of different random wieghts and biases. After all the whole goal of this project is building my ML understanding, not just speed with random functions in libraries.

I've been using this really cool browser extension lately for doing math. The `\partial` command has come in handy every day constantly. I've found once I understood the math and solved derivitives myself, things started making a lot more sense.

I kept looking at these solutions using "delta" all the while thinking how it's strange how I'm doing diff xi and diff for d_w and d_b instead of just setting diff as a universal term or something. It just so happened that I glanced at a solution with calculations for delta and realized it's just the first and second derivitive.

$$
\delta_j = \hat{y}_j - y_j
$$

I then calculated these manually for the second layer for learning purposes.

$$
\frac{\partial L}{\partial W^{(2)}_{h j}} = \delta_j \, a^{(1)}_h
\qquad
\frac{\partial L}{\partial b^{(2)}_j} = \delta_j
$$

$$
\delta^{(1)}_h
= \left( \sum_j W^{(2)}_{h j} \, \delta_j \right) \text{ReLU}'(z^{(1)}_h)
$$
