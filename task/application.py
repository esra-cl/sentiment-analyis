from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from grouping_using_cos_sim import grouping_
import pandas as pd 
class app:
    def __init__(self) -> None:
        self.mlm_model_pth= fr"mlm_model"
        self.llm_model_pth= fr"T5_llm_model"
        self.grouped_df= None

    def load_mlm_model( self,input):
        _model = AutoModelForSequenceClassification.from_pretrained(self.mlm_model_pth)
        _tokenizer = AutoTokenizer.from_pretrained(self.mlm_model_pth)
        inputs = _tokenizer(input, padding=True, truncation=True, return_tensors="pt")
        outputs = _model(**inputs)
        logits = outputs.logits
        predictions = logits.argmax(dim=-1)
        label_mapping = {0: 'Claim', 1: 'Counterclaim', 2: 'Evidence', 3: 'Rebuttal'}
        predicted_label = label_mapping[predictions.item()]
        # print(predicted_label)
        return predicted_label
    

    def load_llm_model(self, input ):
        _model = T5ForConditionalGeneration.from_pretrained(self.llm_model_pth)
        _tokenizer = T5Tokenizer.from_pretrained(self.llm_model_pth)
        inputs = _tokenizer("summarize: " + input, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = _model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
        conclusion = _tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return conclusion

    def grouping( self,topic_):
        g = grouping_(topic=topic_)
        opinion_list = g.group_()
        if len(opinion_list)>30:
            return opinion_list [:30]
        return opinion_list
    
    def parsing_data ( self,event ):
        # {
        # "event_type": "new_topic",
        # "data": {
        #     "topic_id": "67890",
        #     "topic_name": "The impact of electric cars on the environment",
        #     "description": "Discussion about how electric vehicles influence the environment, including both positive and negative aspects.",
        #     "created_at": "2024-08-19T12:00:00Z"
        # }
        # }
        if event["event_type"] == "new_topic":
            opinions = pd.load_csv("dataset/opinions.csv")
            data = event["data"]


    def compute_(self,topic):
        out_put = f"Ana Görüş: {topic}\n"
        ctr=0
        opnions_dic= {}
        print("grouping opinions ---->")
        opnions = self.grouping(topic)
        for  opinion in enumerate(opnions) : 
            input_ = str(topic) + " " + str(opinion)
            output = self.load_mlm_model(str ( input_))
            if output not in opnions_dic:
                opnions_dic[output] = []
            opnions_dic[f"{output}"].append(opinion)
        # print(len(opnions_dic))
        print("-- DONE!")
        print("calssifying the opinions ---->")
        conclusion = self.load_llm_model(str(topic))
        # print("Ana Görüş: ",topic,"\n")
        for key, items in opnions_dic.items():
            for indx , item in enumerate(items) :
                ctr+=1
                out_put = out_put + f"Alakalı Görüş {ctr} ({key})- {item[1]}\n"
                # print(f"Alakalı Görüş {ctr} ({key})-",item[1])
        # print("sonuç:")
        out_put= out_put + f"sonuç: {conclusion}"
        return out_put
# ob = app()
# out_put = ob.compute_("On my perspective, I think that the face is a natural landform because I dont think that there is any life on Mars. In these next few paragraphs, I'll be talking about how I think that is is a natural landform")
