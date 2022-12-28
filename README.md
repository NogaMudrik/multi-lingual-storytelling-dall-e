to install, please visit https://pypi.org/project/multi-lingual-storytelling-dall-e/ (!pip install multi-lingual-storytelling-dall-e)

# multi-lingual-storytelling-dall-e

Mudrik, N., Charles, A., “Multi-Lingual DALL-E Storytime”. Arxiv. (2022). https://arxiv.org/abs/2212.11985

While recent advancements in artificial intelligence (AI) language models demonstrate cutting-edge performance when working with English texts, equivalent models do not exist in other languages or do not reach the same performance level. This undesired effect of AI advancements increases the gap between access to new technology from different populations across the world. This unsought bias mainly discriminates against individuals whose English skills are less developed, e.g., non-English speakers children. Following significant advancements in AI research in recent years, OpenAI has recently presented DALL-E: a powerful tool for creating images based on English text prompts. While DALL-E is a promising tool for many applications, its decreased performance when given input in a different language, limits its audience and deepens the gap between populations. An additional limitation of the current DALL-E model is that it only allows for the creation of a few images in response to a given input prompt, rather than a series of consecutive coherent frames that tell a story or describe a process that changes over time. Here, we present an easy-to-use automatic DALL-E storytelling framework that leverages the existing DALL-E model to enable fast and coherent visualizations of non-English songs and stories, pushing the limit of the one-step-at-a-time option DALL-E currently offers. We show that our framework is able to effectively visualize stories from non-English texts and portray the changes in the plot over time. It is also able to create a narrative and maintain interpretable changes in the description across frames. Additionally, our framework offers users the ability to specify constraints on the story elements, such as a specific location or context, and to maintain a consistent style throughout the visualization.


### How to use?
#### one time pre-steps:
 1) create an account in openai and store your api key. You will have to type your key later.
 2) create a free Google workspace account, create credetials, and download the associated json file. (https://cloud.google.com/apis)
 3) rename the json file to "translation.json"

#### using the package:
1) pip install the package (!pip install multi-lingual-storytelling-dall-e)
2) import the package
3) run song2images([name of text file], parameters)
4) You will find the set of images in the folder 'images' under the current directory
5) in order to create a figure of subplots, run _create_subplots_fig(path of saved images, num_columns (int), path_save = [where to save the subplots fig?], title = [name of file? without .png]) _
6) in order to save to gif, run  _create_gif(path of images, name_save = [how to call the gif?], path_save = [where to save the gif?])_
