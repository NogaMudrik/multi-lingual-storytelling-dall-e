to install, please visit https://pypi.org/project/multi-lingual-storytelling-dall-e/ (!pip install multi-lingual-storytelling-dall-e)

# multi-lingual-storytelling-dall-e

Mudrik, N., Charles, A., “Multi-Lingual DALL-E Storytime”. Arxiv. (2022). https://arxiv.org/abs/2212.11985

Visualizations are a vital tool in the process of
education, playing a critical role in helping individuals com-
prehend and retain information. With the recent advancements
in artificial intelligence and automatic visualization tools, such
as OpenAI’s DALL-E, the ability to generate images based
on text prompts has been greatly improved. However, these
advancements present a significant challenge for populations
with limited English proficiency, exacerbating the educational
divide between children from different backgrounds and limiting
their access to new technology. Here, we introduce a DALL-E
storytelling framework designed to facilitate the fast and coherent
visualization of non-English songs, stories, and biblical texts.
Our framework extends the original DALL-E model to handle
non-English input and allows users to specify constraints on
story elements, such as a specific location or context. The key
advantage of our framework over manual editing of DALL-E
images is that it offers a more seamless and intuitive experience
for the user, as well as automates the process, thus eliminating the
time-consuming and technical-expertise-requiring manual editing
process. The visualization masks are automatically adjusted to
form a coherent story, ensuring that the figures and objects in
each frame are consistent and maintain their meaning throughout
the visualization, allowing for a much smoother experience for
the viewer. Our results demonstrate that our framework is
capable of effectively and quickly visualizing stories in a coherent
way, conveying changes in the plot over time, and creating a
narrative with a consistent style throughout the visualization. By
enabling the visualization of non-English texts, our framework
helps bridge the gap between populations and promotes equal
access to technology and education, particularly for children and
individuals who struggle with understanding complex narrative
texts, such as fast-paced songs and biblical stories. This holds
the potential to greatly enhance literacy and foster a deeper
understanding of these important texts.


### How to use?
#### one time pre-steps:
 1) create an account in openai and store your api key. You will have to type your key later.
 2) create a free Google workspace account, create credetials, and download the associated json file. (https://cloud.google.com/apis)
 3) rename the json file to "translation.json"

#### using the package:
1) pip install the package (!pip install multi-lingual-storytelling-dall-e , see package here https://pypi.org/project/multi-lingual-storytelling-dall-e/)
2) import the package
3) run song2images([name of text file], parameters)
4) You will find the set of images in the folder 'images' under the current directory
5) in order to create a figure of subplots, run _create_subplots_fig(path of saved images, num_columns (int), path_save = [where to save the subplots fig?], title = [name of file? without .png]) _
6) in order to save to gif, run  _create_gif(path of images, name_save = [how to call the gif?], path_save = [where to save the gif?])_
