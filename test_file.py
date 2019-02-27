import pickle
import fasttext
import sys
import json
from numpy import array



#load pre-trained fasttext bin model
model = fasttext.load_model('./data/fasttext_vec20_model.bin')

# load pre-trained incall model
filename_incall = './data/incall_model.sav'
loaded_model_incall = pickle.load(open(filename_incall, 'rb'))

# load pre-trained outcall model
filename_outcall = './data/outcall_model.sav'
loaded_model_outcall = pickle.load(open(filename_outcall, 'rb'))

# load pre-trained movement model
filename_movement = './data/movement_model.sav'
loaded_model_movement = pickle.load(open(filename_movement, 'rb'))

# load pre-trained multi model
filename_multi = './data/multi_model.sav'
loaded_model_multi = pickle.load(open(filename_multi, 'rb'))

# load pre-trained risky model
filename_risky = './data/risky_model.sav'
loaded_model_risky = pickle.load(open(filename_risky, 'rb'))



def incall_predict_prob(vector_array):
    dic = {}
    dic['value'] = "incall"
    dic['score'] = float(loaded_model_incall.predict_proba(vector_array)[:, -1])
    #print  float(loaded_model_incall.predict_proba(vector_array)[:, 0])
    return dic

def outcall_predict_prob(vector_array):
    dic = {}
    dic['value'] = "outcall"
    dic['score'] = float(loaded_model_outcall.predict_proba(vector_array)[:, -1])
    return dic


def movement_predict_prob(vector_array):
    dic = {}
    dic['value'] = "movement"
    dic['score'] = float(loaded_model_movement.predict_proba(vector_array)[:, -1])
    return dic

def multi_predict_prob(vector_array):
    dic = {}
    dic['value'] = "multi_girls"
    dic['score'] = float(loaded_model_multi.predict_proba(vector_array)[:, -1])
    return dic


def risky_predict_prob(vector_array):
    dic = {}
    dic['value'] = "risky_activity"
    dic['score'] = float(loaded_model_risky.predict_proba(vector_array)[:, -1])
    return dic


def rawInputTest():
    x = raw_input("Input: ")
    return x


if __name__ == "__main__":

    input_file = sys.argv[1]   
    output_file = open(sys.argv[2], 'w')
    with open(input_file, 'r') as f:
        for sentence in f:
            text_vector = array(model[sentence])  #get sentence vector from fasttext model
            vector_array = text_vector.reshape(1, -1)
            d = {}

            #d["incall"] = '{:.5f}'.format(incall_predict_prob(vector_array)["score"]) #convert scientific notation to decimal 
            d["incall"] = incall_predict_prob(vector_array)["score"] 
            d["outcall"] = outcall_predict_prob(vector_array)["score"]
            d["movement"] = movement_predict_prob(vector_array)["score"]
            d["multi_grils"] = multi_predict_prob(vector_array)["score"]
            d["risky"] = risky_predict_prob(vector_array)["score"]

            output_file.write("%s " % sentence.strip('\n'))
            output_file.write(json.dumps(d))
            output_file.write('\n')


    f.close()
    output_file.close()








