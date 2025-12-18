def generate_explanation(input_data, prediction):
    reasons = []

    credit_history = input_data.get("Credit_History", 0)
    income = input_data.get("ApplicantIncome", 0)

    if prediction == 1:
        # Approval reasons
        if credit_history == 1:
            reasons.append("good credit history")
        if income >= 4000:
            reasons.append("sufficient income")

        if not reasons:
            reasons.append("overall eligibility criteria")

        return "Loan Approved because the applicant has " + " and ".join(reasons) + "."

    else:
        # Rejection reasons
        if credit_history == 0:
            reasons.append("poor credit history")
        if income < 4000:
            reasons.append("insufficient income")

        return "Loan Rejected because the applicant has " + " and ".join(reasons) + "."
