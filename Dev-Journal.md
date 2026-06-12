<h4 style="margin-bottom:0;">First Hours into Learning Rust</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-11 · Rust Learning Day 1</em>
</p>

> Lot's of confusion but my focus has been to avoid LLM code like it's the plague. Autocorrect is off and so is Copilot.
>
> I'm reading through the amazing Rust documentation along with doing Leatcode problems.
>
> As for the use of AI, I've been using Perplexity.ai for choosing Leatcode problems and writing tests for them exlcusively.
>
> All code is human written. I've loved the look of Rust code and I always imitate it in Python anyways so I'm hoping this keeps up. It's a little bit of a learning curve.
> As long as I keep challenging myself and going through the docs I'll eventually learn.
>
> The syntax too is starting to make sense but I have to think about it.
>
> ```Rust
> Vec<i32> // Right cos Vector = Array = List and i32 interger 32 bits
> for i in 0..nums.len() // 0..nums like the ... in math and .len() is an improvement
> -> Vec<i32> // is also phenominal because no typing library is required
> return // Early return and auto returning is amazing
> ```
>
> Still wrapping my head around som errors with pointers and borrowing. Anyways did two Leatcode problems in rust.
>
> ```Rust
> impl Solution {
>     pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
>         for i in 0..nums.len() {
>             for j in (i+1)..nums.len() {
>                 if nums[i] + nums[j] == target {
>                     return vec![i as i32, j as i32];
>                 }
>             }
>         }
>     }
> }
>
> // Which I optimized using hashmaps to
> use std::collections::HashMap;
> let mut map: HashMap<i32, usize> = HashMap::new();
>
> for (i, &val) in nums.iter().enumerate() {
>     let mut complement = target - val;
>     if let Some(&j) = map.get(&complement) {
>         return vec![j as i32, i as i32];
>     }
>     else {
>         map.insert(val, i);
>     }
> }
>
>
> // &
>
> impl Solution {
>     pub fn is_palindrome(s: String) -> bool {
>         let filtered: String = s
>             .to_lowercase()
>             .chars()
>             .filter(|c| c.is_alphanumeric())
>             .collect();
>
>         let reversed: String = filtered
>             .chars()
>             .rev()
>             .collect();
>
>         filtered == reversed
>     }
> }
> ```

<h4 style="margin-bottom:0;">Steping it up—Leatcode Medium Difficulty</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-11 · Rust Learning Day 2</em>
</p>

Today I worked on a medium difficulty leatcode problem. It went pretty well but I learned some new syntax. This was a command I used quite a lot in Python but getting familiar with it in Rust was nice.

```Rust
let x = 3
let y = 4
x.max(y)
// Output: 4
```

I had already solved the Leatcode problem beforehand but the solution used max and it greatly simplified my code.

I'm also noticing that writing in Rust has already started becoming very natural to me. It's a lot like python in certain ways at least the way I write my Python code—with a little too much perfectionism.

<h4 style="margin-bottom:0;">Notes pre-writing</h4>
<p style="margin-top:0; font-size:0.85em;">
  <em>2026-06-12 · Rust Learning Day 3</em>
</p>

```Rust
// ❌let mut log: Vec<i32> = vec![]; because idx will be usize?!?!
// so instead:
let mut log: Vec<usize> = vec![];
// ...
for (idx, x)  in nums.iter().skip(1).enumerate() {
  // ...
  log.push(idx + 1); // now works
}
```

Then I'm starting to learn about the borrowing system which feels confusing right now but I'm already starting to see the logic behind it.

```Rust
let mut current: i32 = nums[0];
for (idx, x)  in nums.iter().skip(1).enumerate() {
  if current < *x {
    // ... prevents errors by adding *x because current is i32 and
    // x is a reference to a value
  }
}
```

But there's a better solution

```Rust
let mut current: i32 = nums[0];
for (idx, &x) in nums.iter().skip(1).enumerate() {
  if current < x {
    // ... Here by borrowing we avoid the missmatched type issue
    // because we borrow from x
  }
}
```

— just a draft until I write this out tonight
