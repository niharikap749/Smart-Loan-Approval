def generate_explanation(input_data, prediction):
    """
    Generates a human-readable explanation
    for loan approval or rejection.
    """

    explanation = []

    if input_data.get("Credit_History", 0) == 1:
        explanation.append("good credit history")
    else:
        explanation.append("poor credit history")

    if input_data.get("ApplicantIncome", 0) > 4000:
        explanation.append("sufficient income")
    else:
        explanation.append("low income")

    if prediction == 1:
        return "Loan Approved because the applicant has " + " and ".join(explanation) + "."
    else:
        return "Loan Rejected due to " + " and ".join(explanation) + "."
