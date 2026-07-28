# Fraud Detection System for Digital Payments and Banking Transactions

## Yeh project kya karta hai (Simple explanation)

Har bank ya digital wallet (JazzCash, Easypaisa, credit cards, etc.) mein
lakhon transactions hoti hain. Inme se bohot kam (1-2%) **fraud** hoti hain
— matlab kisi ne chori shuda card ya account se paisa nikalne ki koshish ki.

Is project ka kaam hai: **Machine Learning model banana jo automatically
bata de ke koi transaction "normal" hai ya "fraud".**

Yeh ek **classification problem** hai — sirf 2 possible outputs:
- `0` = Legit (theek) transaction
- `1` = Fraud transaction

---

## Dataset (data kahan se aaya)

Real bank data private hota hai, isliye humne **synthetic dataset**
generate kiya hai (`generate_data.py` file) jo bilkul real fraud data jaisa
behave karta hai:

| Feature | Matlab |
|---|---|
| `amount` | Transaction ki rakam |
| `transaction_hour` | Din ke kis waqt transaction hui (0-23) |
| `account_age_days` | Account kitne din purana hai |
| `num_transactions_last_24h` | Pichle 24 ghanton mein kitni transactions hui |
| `distance_from_home_km` | Ghar se kitni door transaction hui |
| `is_foreign_transaction` | Kya transaction videsh se hui (0/1) |
| `is_online` | Kya online transaction thi (0/1) |
| `class` | Target: 0 = legit, 1 = fraud |

**Fraud pattern jo humne data mein dala hai** (real duniya jaisa hi):
fraud transactions aksar raat ko hoti hain, bara amount ki hoti hain,
naye account se hoti hain, aur ghar se bohot door hoti hain.

---

## Kaam kaise hota hai (Pipeline)

1. **Data load karna** — CSV file se transactions read karte hain
2. **Preprocessing** — features ko scale karte hain (StandardScaler) taake
   sab features same range mein aa jayein
3. **Train/Test split** — 75% data se model train karte hain, 25% se test
   (verify) karte hain ke model sahi kaam kar raha hai ya nahi
4. **Do models train karte hain:**
   - **Logistic Regression** — simple, fast, baseline model
   - **Random Forest** — kayi "decision trees" ka combination, zyada
     accurate hota hai
5. **Evaluation** — sirf accuracy dekhna kaafi nahi (kyunke 98% data legit
   hai, koi bhi "sab legit hai" bol kar bhi 98% accuracy pa sakta hai).
   Isliye hum dekhte hain:
   - **Precision** — jab model "fraud" bole, kitni baar sahi hota hai
   - **Recall** — total fraud cases mein se kitne pakre gaye
   - **ROC-AUC** — overall model kitna acha differentiate karta hai
6. **Feature importance** — dekhte hain ke kaunsi cheez (waqt, distance,
   amount) sabse zyada fraud detect karne mein madad karti hai

---

## Kaise run karein

```bash
pip install -r requirements.txt
python generate_data.py       # dataset banata hai (data/transactions.csv)
python fraud_detection.py     # model train + evaluate karta hai
```

Output:
- Terminal mein: dataset stats, dono models ke precision/recall/F1/ROC-AUC
- `results/evaluation_plots.png` — Confusion Matrix + ROC Curve
- `results/feature_importance.png` — kaunsa feature sabse important hai

---

## Results ka matlab (presentation ke liye)

- **Confusion Matrix** dikhata hai: model ne kitni transactions sahi
  pakri (True Positive), kitni ghalat "fraud" bol di (False Positive),
  aur kitni fraud miss ki (False Negative — yeh sabse khatarnak hai
  kyunke asli fraud pakra nahi gaya).
- **ROC Curve**: jitna curve top-left corner ke qareeb, model utna acha.
  AUC = 1.0 matlab perfect model.
- **Feature Importance chart**: batata hai ke model ne fraud pakarne ke
  liye kaunse features sabse zyada use kiye (jaise humare data mein
  `transaction_hour` aur `distance_from_home_km` sabse important nikle
  — jo makes sense hai kyunke fraud aksar raat ko aur ghar se door hota hai).

---

## Agar viva/presentation mein poocha jaye

**Q: Ye supervised ya unsupervised learning hai?**
A: Supervised — kyunke humare paas labeled data hai (har transaction ko
pehle se "fraud" ya "legit" mark kiya gaya hai).

**Q: Class imbalance kya hoti hai aur isse kaise handle kiya?**
A: Jab ek class (legit) dusri se bohot zyada ho (98% vs 2%), tab model
sirf majority class predict karke bhi high accuracy pa sakta hai. Humne
`class_weight="balanced"` use kiya jo model ko fraud cases par zyada
dhyaan dene par majboor karta hai.

**Q: Random Forest, Logistic Regression se better kyun hai (aksar)?**
A: Random Forest kayi decision trees banata hai aur unka average leta
hai, isliye complex, non-linear patterns better pakarta hai. Logistic
Regression simple linear relationships ke liye acha hai.

**Q: Real duniya mein isko improve kaise karenge?**
A: Real fraud data use karke, aur zyada features add karke (jaise
device fingerprint, IP address, spending history pattern), aur models
jaise XGBoost ya Neural Networks try karke.
