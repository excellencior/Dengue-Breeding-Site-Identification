## Metrics Table

| Metric | Value |
|------------------------------|--------------------------|
| construction_site_prob | 0.016835016835016835 |
| construction_site_avg | 0.06397306397306397 |
| flower_pot_prob | 0.531986531986532 |
| flower_pot_avg | 8.653198653198654 |
| open_tank_prob | 0.037037037037037035 |
| open_tank_avg | 0.050505050505050504 |
| polythene_prob | 0.05723905723905724 |
| polythene_avg | 0.09090909090909091 |
| reservoir_prob | 0.006734006734006734 |
| reservoir_avg | 0.013468013468013467 |
| tyres_prob | 0.02356902356902357 |
| tyres_avg | 0.02356902356902357 |
| expected_objects_per_building | 8.895622895622896 |
| risk_probability | 0.8703401091157507 |

### Risk Probability Explanation

The **87.03% risk probability** represents the overall likelihood that **any given building containing at least one object** is risky.
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
