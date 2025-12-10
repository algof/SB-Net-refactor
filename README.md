# SB-Net

## 🛠 Installation & Software Usage Guide  

### Prerequisites  
Before running this application, make sure you have installed the following dependencies:
- pandas
- numpy
- matplotlib
- sklearn

### Steps
1. **Prepare the Dataset**  
   - Download the datasets:
      - CTU-13
      - NCC
      - NCC-2
   - Place all datasets into the `/Datasets/` folder following the structure and guidelines described in `/Datasets/README.txt`.

2. **Data Preprocessing**
	-	Run the Jupyter Notebook script `data_maker_{dataset}.ipynb` for each dataset to generate the train and test files.
	-	Combine all `train` files by running `train_combiner.py`.
   
3. **Data Splitting**
	-	Execute `combined_train_test_maker.ipynb` to split `combined_train.csv` into `train` and `test` datasets ready for use.

4. **Ensamble Feature Selection**
	-	Run the script `rank_aggragation.py` to perform ensemble feature selection using multiple methods.
	-	The feature ranking results will be used by `borda_score.py` to aggregate ranking scores into a final result.

5. **Cascade Learner Classification**
	-	Step 1: Run `looping_classification.ipynb` to find the best algorithm combination for the two-stage cascade classification.
	-	Step 2: Run `looping_features.ipynb` to determine the optimal number of features based on classification performance.

6. **Evaluasi Model**
	-	Finally, run `final_classification_test.py` to test the model’s performance on the prepared test data.
