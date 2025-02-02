## Project Overview
This Python script applies the **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)** method to rank **pre-trained text generation models** based on their performance metrics.  
It evaluates models like **GPT-3, LLaMA-2, BART, T5, and GPT-2** using key metrics such as **Perplexity, BLEU, ROUGE, Inference Time, and Model Size**.  

The script also generates **visualizations** to help interpret the rankings.

## Features
 **Implements TOPSIS** to rank text generation models  
 **Uses multiple evaluation metrics** (Perplexity, BLEU, ROUGE, etc.)  
 **Generates visualizations**:
- **Bar Chart**: Displays TOPSIS scores  
- **Scatter Plot**: Shows Perplexity vs. BLEU  
- **Radar Chart**: Compares all metrics for each model  

## **Installation**
Ensure you have the following Python packages installed:

```bash
pip install numpy pandas matplotlib seaborn
```
---

## **How It Works**


### **Normalize the Data**
- **Benefit Criteria:** Higher values are better (**BLEU, ROUGE**)  
- **Cost Criteria:** Lower values are better (**Perplexity, Inference Time, Model Size**)  

Normalization ensures that all values are **comparable**.

---

### **Compute Ideal Best & Worst Values**
- The **Ideal Best** is the best value for each metric.
- The **Ideal Worst** is the worst value for each metric.

---

### **4Compute TOPSIS Score**
- Calculate **distances** from the ideal best and worst solutions.
- Compute **TOPSIS score** for each model.

**Final Ranking (Example Output):**

| Model   | TOPSIS Score | Rank |
|---------|-------------|------|
| **BART**  | 0.778       | 1    |
| **LLaMA-2** | 0.592       | 2    |
| **T5**   | 0.583       | 3    |
| **GPT-3**  | 0.536       | 4    |
| **GPT-2**  | 0.000       | 5    |

---

## **Visualizations**
📊 **Bar Chart** → TOPSIS score comparison  
📈 **Scatter Plot** → BLEU vs. Perplexity  
🕸 **Radar Chart** → Multi-metric comparison  

---

## **How to Run the Code**
Simply run the script in a **Python environment** (Jupyter Notebook, VS Code, etc.):

```python
python topsis_model_selection.py
```

---

## **Customization**
🔹 **Use Real Data**: Replace the dummy dataset with actual model evaluation results.  
🔹 **Change Weights**: Modify the `weights` array to assign different importance to each metric.  
🔹 **Add More Models**: Extend the dataset by adding new models and their respective scores.  
