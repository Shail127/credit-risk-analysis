Credit Risk Analysis — ML Approach to Loan Approval & Default Prediction

Multiclass ML system predicting loan outcomes (Approved / Pending / Rejected) from 5,000 borrower records, using 46 demographic, financial, and engineered features (DTI, EMI burden, composite 300–900 credit score).


-Compared XGBoost, LightGBM, Random Forest, and a Backpropagation Neural Network
-Best model: XGBoost — 73.8% accuracy, 0.876 macro AUC after heuristic hyperparameter tuning
-Built interactive customer-level risk lookup tools (DTI ratio, credit score, risk category)
-Stack: Python, scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, Pandas
