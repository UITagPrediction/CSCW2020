import argparse
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from loader import data_loader
import pickle

parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--tag', type=str, default='blue',
                    help='model to train (default: blue)')
parser.add_argument('--classifer', type=str, default='svm',
                    help='binary classifer [svm, dt] (default: svm)')
parser.add_argument('--feature', type=str, default='hist',
                    help='image feature [hist, hsv, sift] (default: hsv)')

args = parser.parse_args()
print(args)

df_train, df_test = data_loader(args.tag,args.feature)
df_train = shuffle(df_train)

def train():
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']
    X_test = df_test.drop('label', axis=1)
    y_test = df_test['label']
    
    # Standardize
    stdScale = StandardScaler().fit(X_train)
    X_train = stdScale.transform(X_train)
    X_test = stdScale.transform(X_test)

    if args.classifer == 'svm':
        from sklearn.svm import SVC
        svclassifier = SVC(kernel='linear')
        print('-'*10)
        print(svclassifier)
        svclassifier.fit(X_train, y_train)
        y_pred = svclassifier.predict(X_test)
        print('-'*10)
        print(confusion_matrix(y_test,y_pred))
        print("Accuarcy: ", accuracy_score(y_test, y_pred))
        print(classification_report(y_test,y_pred))
    else:
        from sklearn import tree
        dtclassifer = tree.DecisionTreeClassifier()
        print('-'*10)
        print(dtclassifer)
        dtclassifer.fit(X_train, y_train)
        y_pred = dtclassifer.predict(X_test)
        print('-'*10)
        print(confusion_matrix(y_test,y_pred))
        print("Accuarcy: ", accuracy_score(y_test, y_pred))
        print(classification_report(y_test,y_pred))

if __name__ == "__main__":
    train()