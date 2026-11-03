
#importing libraries
import customtkinter
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import string
import contractions
import torch
import numpy as np
from tkinter import messagebox

#aspects dictionary for clarity output
aspects={
    'account-management.account-access' : "Managing Account Access",
    'company-brand.general-satisfaction': "Brand Satisfaction",
    'company-brand.competitor': "Brand Competitor Comparison",
    'company-brand.reviews': "Customer Reviews",
    'logistics-rides.speed': "Ride Speed",
    'online-experience.app-website': "App & Website Experience",
    'purchase-booking-experience.ease-of-use': "Ease of Purchasing/Booking Experience",
    'staff-support.attitude-of-staff': "Staff Attitude",
    'staff-support.phone': "Call Support",
    'staff-support.email': "Email Support",
    'value.price-value-for-money': "Prices",
    'value.discounts-promotion':"Discounts & Promotions"
}

recommendations = {

    "App & Website Experience":
    "Review and enhance the application's design and user interface.",

    "Ride Speed":
    "Improve system responsiveness and overall performance.",

    "Ease of Purchasing/Booking Experience":
    "Maintain the current user-friendly booking experience.",

    "Prices":
    "Review pricing strategy and customer value perception.",

    "Staff Attitude":
    "Continue staff training and customer service initiatives.",

    "Call Support":
    "Improve support response time and customer communication.",

    "Email Support":
    "Improve email response quality and turnaround time.",

    "Managing Account Access":
    "Maintain account accessibility and authentication reliability.",

    "Brand Satisfaction":
    "Continue maintaining overall customer satisfaction levels.",

    "Discounts & Promotions":
    "Review promotional offerings and customer incentives.",

    "Customer Reviews":
    "Monitor customer feedback trends and address recurring concerns."
}

#aspect labels construction
aspects_plain=[
        'account-management.account-access',
        'company-brand.general-satisfaction',
        'company-brand.competitor',
        'company-brand.reviews',
        'logistics-rides.speed',
        'online-experience.app-website',
        'purchase-booking-experience.ease-of-use',
        'staff-support.attitude-of-staff',
        'staff-support.phone',
        'staff-support.email',
        'value.price-value-for-money',
        'value.discounts-promotion'
    ]

sentiments=[-1,0,1]

#creating all the aspect labels
total_labels=[]
for aspect in aspects_plain:
    for sent in sentiments:
        total_labels.append(f"{aspect}.{sent}")

#should come from bert model
tokenizer=AutoTokenizer.from_pretrained("deberta_absa_model")
prediction_model=AutoModelForSequenceClassification.from_pretrained("deberta_absa_model")
prediction_model.eval()
   

#predicting the aspects and sentiments from user input
def get_predictions():
   
   #get user input
   text=input.get("0.0", "end").strip()
   
   #tokenize the text input
   inputs=tokenizer(text, truncation=True, return_tensors="pt",padding=True, max_length=128)

   #get outputs from deberta
   with torch.no_grad():
      outputs=prediction_model(**inputs)

   #get the probabilities
   probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

   sorted_probs = sorted(
      zip(total_labels, probs),
      key=lambda x: x[1],
      reverse=True
   )



   for label, prob in sorted_probs[:10]:
      print(f"{label} --> {round(float(prob),3)}")


   print("\n")

   predictions = []
   prediction_probs = []


   for label, prob in sorted_probs[:5]:

      if prob >= 0.25:
         predictions.append(label)
         prediction_probs.append(prob)

   highest_confidence = max(prediction_probs) if prediction_probs else 0

   output =  "\n"
   
   # Group predictions by aspect
   aspect_results = {}

   for prediction, prob in zip(predictions, prediction_probs):

      aspect, sentiment = prediction.rsplit(".", 1)

      if aspect not in aspect_results:
         aspect_results[aspect] = {}

      aspect_results[aspect][int(sentiment)] = float(prob)

   output += "📊 ANALYSIS RESULTS\n"
   output +=  "\n\n"


   positive_count = 0
   negative_count = 0
   neutral_count = 0
   mixed_count = 0

   positive_aspects = []
   negative_aspects = []
   mixed_aspects = []
   neutral_aspects = []

   output += "🔍 Detected Aspect Sentiments:\n\n"

   for aspect, sentiment_scores in aspect_results.items():

      display_aspect = aspects.get(aspect, aspect)

      positive_prob = sentiment_scores.get(1, 0)
      neutral_prob = sentiment_scores.get(0, 0)
      negative_prob = sentiment_scores.get(-1, 0)

      positive_conf = round(positive_prob * 100, 1)
      neutral_conf = round(neutral_prob * 100, 1)
      negative_conf = round(negative_prob * 100, 1)

      # Mixed
      if positive_prob > 0.25 and negative_prob > 0.25:

         mixed_count = mixed_count + 1

         icon = "⚖"

         mixed_aspects.append(display_aspect)

         output += f"{icon} {display_aspect}\n"
         output += f"  Sentiment: Mixed\n"
         output += f"  Positive Confidence: {positive_conf}%\n"
         output += f"  Negative Confidence: {negative_conf}%\n\n"

      # Positive
      elif positive_prob >= max(neutral_prob, negative_prob):

         icon = "✓"

         output += f"{icon} {display_aspect}\n"
         output += f"  Sentiment: Positive\n"
         output += f"  Confidence: {positive_conf}%\n\n"

         positive_count += 1
         positive_aspects.append(display_aspect)

      # Negative
      elif negative_prob >= max(positive_prob, neutral_prob):

         icon = "⚠"

         output += f"{icon} {display_aspect}\n"
         output += f"  Sentiment: Negative\n"
         output += f"  Confidence: {negative_conf}%\n\n"

         negative_count += 1
         negative_aspects.append(display_aspect)

      # Neutral
      else:

         icon = "•"

         output += f"{icon} {display_aspect}\n"
         output += f"  Sentiment: Neutral\n"
         output += f"  Confidence: {neutral_conf}%\n\n"

         neutral_count += 1
         neutral_aspects.append(display_aspect)

   if positive_count > negative_count and mixed_count == 0:
      overall_sentiment = "POSITIVE"

   elif negative_count > positive_count and mixed_count == 0:
      overall_sentiment = "NEGATIVE"

   elif mixed_count > 0:
      overall_sentiment = "MIXED"

   else:
      overall_sentiment = "NEUTRAL"

   output = (
    f"📊 ANALYSIS RESULTS\n\n"
    f"Overall Sentiment: {overall_sentiment}\n\n"
    + output
)



   output += "."*50 + "\n\n"

   output += "📝 Executive Summary:\n\n"

   if positive_aspects:

      output += (
         "Customers expressed satisfaction regarding "
         + ", ".join(positive_aspects)
         + ".\n\n"
      )

   if negative_aspects:

      output += (
         "Areas requiring improvement were identified in "
         + ", ".join(negative_aspects)
         + ".\n\n"
      )

   if mixed_aspects:

      output += (
         "Customers reported both positive and negative experiences related to "
         + ", ".join(mixed_aspects)
         + ".\n\n"
      )

   if neutral_aspects:

      output += (
         "Neutral feedback was recorded for "
         + ", ".join(neutral_aspects)
         + ".\n\n"
      )

   if highest_confidence < 0.60:

      output += (
         "Analysis Note:\n"
         "The review contains limited context. "
         "Aspect predictions may be less reliable.\n\n"
      )

   output += "🎯 Recommended Actions:\n\n"

   for aspect in negative_aspects:

      if aspect in recommendations:
         output += f"• {recommendations[aspect]}\n"

   for aspect in mixed_aspects:

      if aspect in recommendations:
         output += (
               f"• Mixed feedback detected for {aspect}. "
               f"{recommendations[aspect]}\n"
         )

   for aspect in positive_aspects:

      if aspect in recommendations:
         output += f"• Continue maintaining: {recommendations[aspect]}\n"

   output += "\n" 


   #place results in label to be displayed
   result_label.configure(state="normal")
   print(output)
   result_label.delete("0.0", "end")
   result_label.insert("0.0", output)
   result_label.configure(state="disabled")

   return predictions
      


def button_callback():
   
   text_input=input.get("0.0", "end").strip()

   if text_input=="":

      result_label.configure(state="normal")
      result_label.delete("0.0", "end")
      result_label.insert("0.0", "Feedback NOT Entered")
      result_label.configure(state="disabled")
      return

   if len(text_input.split()) < 4:

      messagebox.showwarning(
         "Insufficient Feedback",
         "Your review is too short for reliable analysis.\n\n"
         "Please provide more detailed feedback."
      )
      return
   
   output_sys.geometry("1000x750")
   get_predictions()

APP_BG = "#1D4438"      # Beige
FRAME_BG = "#21483D"    # White
PRIMARY = "#2F5D50"     # Dark Green
SECONDARY = "#4C7A6A"   # Lighter Green
TEXT = "#1E1E1E"

output_sys=customtkinter.CTk()
output_sys.configure(fg_color="#1D4438")
output_sys.geometry("1000x750")
output_sys.title("Aspect-Based Sentiment Analysis System")

title_label = customtkinter.CTkLabel(
    output_sys,
    text="Customer Feedback Analysis",
    font=("Segoe UI", 24, "bold"),
    text_color="#F5F0E6"
)
title_label.pack(pady=(20,10))

subtitle_label = customtkinter.CTkLabel(
    output_sys,
    text="Aspect-Based Sentiment Analysis Platform",
    font=("Segoe UI", 14),
    text_color="#EADFCB"
)
subtitle_label.pack()

input = customtkinter.CTkTextbox(
    output_sys,
    width=700,
    height=150,
    corner_radius=15,
    font=("Bahnschrift", 15),
    fg_color="#F5F0E6",
    text_color="#1E1E1E",
    border_width=2,
    border_color="#D9C7A2"
)
input.pack(padx=20, pady=20)
text=input.get("0.0","end")


enter_button = customtkinter.CTkButton(
    output_sys,
    text="Analyze Review",
    width=200,
    height=40,
    corner_radius=10,
    fg_color="#D9C7A2",
    hover_color="#C9B48A",
    text_color="#1D4438",
    font=("Calibri", 15, "bold"),
    command=button_callback
)
enter_button.pack(padx=20, pady=20)

results_header = customtkinter.CTkLabel(
    output_sys,
    text="Analysis Results",
    font=("Segoe UI", 18, "bold"),
    text_color="#D5F0E9"
)
results_header.pack(pady=(20,5))

result_label = customtkinter.CTkTextbox(
    output_sys,
    width=850,
    height=320,
    corner_radius=15,
    font=("Bahnschrift", 15),
    fg_color="#F5F0E6",
    text_color="#1E1E1E",
    border_width=2,
    border_color="#D9C7A2"
)
result_label.pack(pady=20)
result_label.configure(state="disabled")

footer = customtkinter.CTkLabel(
    output_sys,
    text="DeBERTa Aspect-Based Sentiment Analysis",
    font=("Segoe UI", 10)
)
footer.pack(pady=10)

output_sys.mainloop()



