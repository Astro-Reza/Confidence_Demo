### 3.3.2 Persamaan Confidence-Weighted Fusion

Confidence score untuk setiap sensor $i$ diberikan oleh:

$$c_i = \frac{1}{1 + \sigma_i^2 + \epsilon_i} \tag{3.1}$$

dengan $\sigma_i^2$ merupakan variansi pengukuran dan $\epsilon_i$ merupakan error 
residual sensor. Nilai $c_i$ berada dalam rentang $[0, 1]$, di mana nilai 
yang mendekati 1 menunjukkan tingkat kepercayaan tinggi.

Quaternion hasil fusi dihitung melalui weighted average ternormalisasi:

$$\mathbf{q}_{fused} = \frac{\sum_{i=1}^{n} c_i \mathbf{q}_i}{\left\| \sum_{i=1}^{n} c_i \mathbf{q}_i \right\|} \tag{3.2}$$

Untuk integrasi temporal dengan kecepatan sudut, quaternion diperbarui 
sesuai persamaan:

$$\mathbf{q}_{fused}(t + \Delta t) = \mathbf{q}_{fused}(t) \otimes \exp\left( \frac{\Delta t}{2} \sum_{i=1}^{n} c_i \boldsymbol{\omega}_i \right) \tag{3.3}$$

di mana $\otimes$ menyatakan perkalian Hamilton dan $\exp(\cdot)$ merupakan 
eksponensial quaternion.