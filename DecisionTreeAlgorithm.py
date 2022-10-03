from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from six import StringIO 
import pydot


breast_cancer = load_breast_cancer()
breast_cancer_features = breast_cancer.data
breast_cancer_target = breast_cancer.target

logistic_classification = LogisticRegression()
logistic_classification.fit(breast_cancer_features, breast_cancer_target)

with open("output_logistic_coefficients.txt", "w+") as output_file:
    for f, w in zip(breast_cancer.feature_names, logistic_classification.coef_[0]):
        output_string = "{0:23}: {1:6.2f}".format(f, w)
        output_file.write(output_string + "\n")
        #Here we have positive and negative coefficients that affect
        #the recognition of cancer. Large positive coefficients affect positive
        #result. Otherwise, large negative coefficients affect negative result.
        
decision_tree_classification = DecisionTreeClassifier(max_depth=2)
decision_tree_classification.fit(breast_cancer_features, breast_cancer_target)
dot_data = StringIO()
export_graphviz(decision_tree_classification, out_file=dot_data)
decision_tree_graph = pydot.graph_from_dot_data(dot_data.getvalue())
decision_tree_graph[0].write_pdf("cancer_decision_tree_graph.pdf")
    
    