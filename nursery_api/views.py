from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from .apps import NurseryApiConfig


# This is a simple api to provide the Bagging_KNN_model for the nursery data to clients
class DetermineAdmissionStatus(APIView):
    def post(self, request):
        request = request.data
        parents = request["parents"]  # What kind of parents?
        has_nurs = request["has_nurs"]  # Do the child's parents have nursery school education?
        children = request["children"]  # How many children have the parents?
        housing = request["housing"]  # What the housing status of the family?
        finance = request["finance"]  # How financially convenient is it for
        # the parent to enroll their child in the school?

        health = request["health"]    # What is the status of the child?
        class_model = NurseryApiConfig.model
        prediction = class_model.predict([[
            parents,
            has_nurs,
            children,
            housing,
            finance,
            health
        ]])

        """
            A prediction of 0 means that the prospective student is not recommended, 
            a prediction of 1 means that the prospective student is recommended without special priority
            while a prediction of 2 means that he/she is recommended with special priority
        """
        admission_status = (
            "Not Recommended" if prediction[0] == 0
            else "Recommended Without Special Priority" if prediction[0] == 1
            else "Recommended With Special Priority"
        )

        response_dict = {"admission_status": admission_status}
        print(response_dict)

        return Response(response_dict)

    def get(self, request):
        template_name = "nursery_api/nursery_api_home.html"
        return render(request, template_name=template_name)


home = DetermineAdmissionStatus.as_view()




