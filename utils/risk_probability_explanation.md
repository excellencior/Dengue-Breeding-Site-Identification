### Risk Probability Explanation

The **risk probability** represents the overall likelihood that **any given building containing at least one object** is risky.

### How the Risk Probability is Calculated

1. **Risk for Each Object Type**:
   Each object type (e.g., flower pots, tyres) contributes to the risk based on:
   - Its **detection accuracy**.
   - The **average number of such objects (n_i)** on buildings.
   For example, if flower pots are present more often and are accurately detected, their contribution to risk increases.

2. **Combined Risk Across Object Types**:
   For each building, the risk combines contributions from all object types. This combination uses the formula:
   
   P_I = 1 - PROD(1 - accuracy_class)^{n_i}
   
   where \( n_i \) is the expected number of objects of type \( i \) on the building.

3. **Risk Across the Dataset**:
   The overall **risk probability** comes from averaging the risk contributions of all buildings based on their specific object counts and types.

### Intuition Behind the Risk Probability

- Buildings with **more types of objects** (or higher counts of certain objects) tend to have higher individual risk probabilities.
- When the contributions of all objects and their distributions are considered, the **average risk probability** across the buildings stabilizes at the calculated value.

### What the Risk Probability Means in Practice

- **If you pick a random building, there’s a high chance it is risky.**
- This doesn't mean every building is exactly as risky; some buildings may be less risky (fewer objects or low-accuracy detections), and others more so.
