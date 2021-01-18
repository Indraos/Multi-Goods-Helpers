# Multi-Goods-Helpers
Utilities to solve and visualize multi-goods monopolist problems and their dual variables.
## Getting started
Run
```bash
pip3 install -r requirements.txt
chmod a+x md_helpers.py
./md_helpers.py
```
in your console to get 20 sample files.
## Example usage for Custom Data
```python
types = np.array([[0,0],[1,1]])
probabilities = np.array([1/2,1/2])
monopolist = MGMProblem(types, probabilities)
monopolist.solve()
monopolist.save("monopolist_viz.html")
```
