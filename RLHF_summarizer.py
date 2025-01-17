from aih_automaton import Agent, Task, LinearSyncPipeline
from aih_automaton.tasks.task_literals import OutputType
from typing import List
from source.AzureOpenai import AzureOpenAIModel
from dotenv import load_dotenv
import os 
load_dotenv()
azure_api_key = os.getenv("API_Key")
azure_endpoint = os.getenv("End_point")
azure_engine = os.getenv("Engine")
azure_api_version = os.getenv("API_version")

azure_model_text = AzureOpenAIModel(
    azure_endpoint=azure_endpoint,
    azure_api_key=azure_api_key,
    azure_api_version=azure_api_version,
    parameters={"model":'gpt-35-turbo'}
    # azure_engine=azure_engine
)

def retry_summary_update(
   summary,
    feedback_from_user,
    previous_feedbacks: List = None,
    additional_context: str = None,
):
    """
    Update the Summary based on feedback provided by the user.
    """
    user_prompt = f"""
    Please do the following updates to the Summary based on the user's feedback:

    **Summary To be updated**:
    {summary}

    **User Feedback**:
    {feedback_from_user}

    { '**Additional Context**:' + additional_context 
    + "Ensure to incorporate this additional context information provided if relevant to the user feedback" 
    if (additional_context is not None) and (additional_context != '')
    else '' }
    
    
    """

    # print('User prompt',user_prompt)
    # exit()
    instructions = f"""
    ""You are a Summary writer Your task is to update the Summary based on user feedbacks.""    
    
    **Instructions for Updating the Blog Content**:
    - You will be provided feedback from the user as a primary source to improve the Summary.
    - Focus on the specific areas of feedback, such as improving engagement, refining tone, or adding missing details.
    - Edit the Summary structure or wording where necessary to better align with the user’s expectations.
    - Ensure that the revised Summary incorporates all suggestions and provides a more refined, user-centered version.


    **Actionable Changes**:
    1. Analyze the feedback and identify key areas that need to be addressed.
    2. Make edits in the introduction, product overview, or any specific section mentioned by the user.
    3. Ensure SEO keywords remain naturally integrated throughout the Summary.
    4. Improve the readability, tone, or call-to-action as per feedback.

    { "Feedbacks previously Given by the User: " + previous_feedbacks + 'Ensure you consider these feedbacks for self-reflection as well as preference of user for updating the response.' if previous_feedbacks  else "" }

    Important Note: 
    - Do not add Greetings or Salutations in the Summary not the closing  note (Remember....etc.) in the end. Just Start with Summary 
    - Ensure the length of Summary remain same.
    - Ensure to The updates are made to Summary is specific to the user feedback, other part of the Summary should not be changed unless it is specified by the User.
    - You will do only what is asked in the feedback.
    - Ensure the content is in proper markdown format  specially the headings and subheadings, and bullet points and nested bullet points,  dont use "-" for subheadings and  for bullet points like this (- **example**).
    - Ensure not to add tabular data in the Summary, instead content should be in paragraphs, line by line format.
"""

    blog_rewriter = Agent(
        role="Summary updater with Human Feedback Agent",
        prompt_persona=user_prompt,
        # prompt_persona=updated_blog_content
    )
    # Define the task for the Blog Composer
    task = Task(
        name="Update Summary by user feedback",
        model=azure_model_text,
        agent=blog_rewriter,
        instructions=instructions,
        output_type=OutputType.TEXT,
        # messages=messages
    )

    # Run the pipeline
    output = LinearSyncPipeline(
        name="Summary updater Pipeline",
        completion_message="Summary Updated",
        tasks=[task],
    ).run()

    # Extract the composed Summary
    Summary = output[0]["task_output"]
    # print(blog_content)
    # input("Press Enter to continue...")
    return Summary