# %%writefile app.py
import streamlit as st
import gdown

# Create a title and subtitle
st.title("DLOPS Demo")
st.subheader("Enter the text to be predicted")

# Create a text input box for the user to enter their name
text = st.text_input("text")

from simpletransformers.ner import NERModel, NERArgs

mtype='xlmroberta'
model_args = NERArgs()
# model_args.num_train_epochs=20
model_args.labels_list = ['O', 'B-geo', 'B-gpe', 'B-per', 'I-geo', 'B-org', 'I-org', 'B-tim', 'I-per', 'I-gpe', 'I-tim']
# model_args.overwrite_output_dir = True
url = "https://drive.google.com/drive/folders/1-6ZGGP1g6BpHwpGDQUS3SD8lAnkz-0gr?usp=share_link"
output = "checkpoint-7803-epoch-9"
gdown.download_folder(url, quiet=False)

model = NERModel(mtype,output,args=model_args,use_cuda=False)
#model = NERModel(mtype,r"/content/drive/MyDrive/Classroom/DLOps (CSP7030) 22-23 Sem II/checkpoint-7803-epoch-9",args=model_args,use_cuda=False)
print("model loaded successfully")
# st.cache

# Check if the user has entered a name
if text:
    # Display a greeting with the user's name
    print("running prediction")
    predictions, raw_outputs = model.predict(["I am travelling to India  and I work for Google", "I am a Microsoft employee"])
    print(predictions[0])
    st.write("entities:")
    #st.write(predictions[0])
