# Orchestrator Approval & Routing Dashboard

Streamlit dashboard for BridgerPay/Orchestrator transaction reports.

## Payment Type Logic
- Confirmo = Crypto
- PayPal = P2P
- All other PSPs = International Card

## Core Metrics
- Approval Ratio based on unique `merchantOrderId`
- Retry Ratio and retried order analysis
- PSP-wise approval ratio
- Country-wise approval ratio
- MID-wise and date-wise performance
- PSP-to-PSP decline reason comparison
- Date-wise decline reason comparison
- Country-wise PSP routing recommendation

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Hosting
1. Upload these files to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Select the repository.
4. Set main file path as `app.py`.
5. Deploy.

## Expected File
Upload your orchestrator CSV report from the dashboard sidebar.

Important columns expected: `pspName`, `country`, `merchantOrderId`, `status`, `declineReason`, `midAlias`, and a date column such as `processing_date`, `processingDate`, or `completionDate`.
