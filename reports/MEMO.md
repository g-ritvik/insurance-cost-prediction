# Regression Model Memo

1. **Did adding `age` improve R² over the `bmi`-only baseline? By how much?**

   Yes, adding `age` improved the R² from 0.0394 for the BMI-only baseline to 0.1173 for the two-feature model. This is an improvement of 0.0779, or 7.79 percentage points.


2. **Do the Normal Equation and Gradient Descent weights agree with each other? Why should they, in theory?**

   Yes, both methods produced the same weights to two decimal places: `w0 = -6437.35`, `w1 = 333.39`, and `w2 = 241.90`. They should agree because both methods minimize the same least-squares objective, although the Normal Equation solves for the coefficients directly while Gradient Descent reaches the solution iteratively.


3. **In plain language, what does the sign and size of `w2` (age) tell a non-technical stakeholder?**

   The age coefficient is positive at `241.90`, meaning that, while holding BMI constant, each additional year of age is associated with approximately 241.90 higher predicted medical expenses. This suggests that age provides additional information about expenses beyond BMI alone.


4. **Is this two-feature model good enough to deploy? Why or why not?**

   The model is not strong enough to deploy as a standalone prediction model because its R² is only 0.1173, meaning it explains about 11.73% of the variation in medical expenses. Although adding age improved the model compared with BMI alone, most of the variation in expenses remains unexplained. Additional features, testing on unseen data, and further model evaluation would be needed before considering deployment.
