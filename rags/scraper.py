import os
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph, SearchGraph, SmartScraperMultiGraph
from scrapegraphai.utils import prettify_exec_info
from pydantic import BaseModel
import sys
load_dotenv()

def convert(struct, indent, res_list, sources):
    if type(struct) == dict:
        for k, v in struct.items():
            if k in {'source', 'sources'}:
                sources.extend(v)
            res_list.append(' '*indent+k+': ')
            if type(v) in {str, int}:
                res_list.append(v+'\n')
            else:
                if type(v) == dict:
                    res_list.append('\n')
                convert(v, indent+1, res_list, sources)
    elif type(struct) == list:
        for idx, i in enumerate(struct):
            if type(i) in {str, int}:
                res_list.append(i)
                res_list.append('\n' if idx == len(struct)-1 else ', ')
            elif type(i) == dict:
                res_list.append('\n')
                convert(i, indent+1, res_list, sources)

openai_key = os.getenv("OPENAI_API_KEY")

graph_config = {
   "llm": {
      "api_key": openai_key,
      "model": "openai/gpt-4o-mini",
   },
}

def perform_web_search(query, media_label):
    prompt = f"""{query} related to {media_label}"""
    # Debugging 
    # temp_result =  "Google_I/O_2024_keynotes_highlights: \n 1: \n  title: Gemini updates\n  details: Gemini 1.5 Pro is now available globally for developers. Gemini Nano is designed for local use for tasks like summarization and 'help me write' functionality.\n 2: \n  title: Search upgrades\n  details: AI-powered overviews in search for one billion users through the Search Generative Experience (SGE).\n 3: \n  title: Android updates\n  details: Gemini replaces Google Assistant as the default AI assistant, with multimodal functions for a better smartphone experience.\n 4: \n  title: Workspace updates\n  details: Google Workspace services will now have Gemini 1.5 Pro as the accompanying LLM.\n 5: \n  title: New launches\n  details: Introduction of Veo, a text-to-video generator, and Imagen 3, a text-to-image generator.\n 6: \n  title: Hardware\n  details: Launch of the Pixel 8A and a new generation of Tensor Processing Unit (TPU) called Trillium.\n 7: \n  title: AI advancements\n  details: The keynote address by Sundar Pichai discusses Google’s AI advancements and their impact on search and other products, highlighting the potential of Gemini to revolutionize the way we search for information and interact with computers.\n 8: \n  title: Multimodal capabilities\n  details: Gemini is a multimodal, large language model that can understand and process information from text, code, images, and video, enabling it to answer complex questions and generate various creative text formats.\n 9: \n  title: YouTube enhancements\n  details: A new feature in YouTube uses large language models to make educational videos more interactive.\n 10: \n  title: Vision for AI\n  details: The keynote emphasizes reimagining how AI can enhance creativity, connectivity, and discovery.\nsources: https://www.spiceworks.com/tech/tech-general/articles/google-io-2024-highlights/, https://www.youtube.com/watch?v=XEzRZ35urlk, https://medium.com/google-developer-experts/google-io-2024-summary-created-with-gemini-39b51d190fbb\n\n\nGoogle_I_O_2024_keynote_highlights: \n 1: \n  title: Gemini updates\n  details: Gemini 1.5 Pro is now available globally for developers. Gemini Nano is designed for local use for tasks like summarization and 'help me write' functionality.\n 2: \n  title: Search upgrades\n  details: AI-powered overviews in search for one billion users through the Search Generative Experience (SGE).\n 3: \n  title: Android updates\n  details: Gemini replaces Google Assistant as the default AI assistant, with multimodal functions integrated into Android.\n 4: \n  title: Workspace updates\n  details: Google Workspace services will now have Gemini 1.5 Pro as the accompanying LLM.\n 5: \n  title: New launches\n  details: Introduction of Veo, a text-to-video generator, and Imagen 3, a text-to-image generator.\n 6: \n  title: Hardware\n  details: Launch of the Pixel 8A and announcement of the sixth-generation Tensor Processing Unit (TPU), Trillium.\n 7: \n  title: AI advancements\n  details: Google I/O ’24 is a keynote address by Sundar Pichai about Google’s AI advancements and their impact on search and other products. The launch of Gemini, a new AI model from Google, is highlighted, showcasing its multimodal capabilities to understand and process information from text, code, images, and video. Gemini is being integrated into many Google products, including Search, Photos, and Workspace, with the potential to revolutionize how we search for information and interact with computers.\n 8: \n  title: Interactive educational content\n  details: A new feature in YouTube uses large language models to make educational videos more interactive.\n 9: \n  title: Vision for AI\n  details: Join us as we reimagine how AI can make your life better and help you explore your creativity, connect with the world, and discover new possibilities.\nsources: https://www.spiceworks.com/tech/tech-general/articles/google-io-2024-highlights/, https://www.youtube.com/watch?v=XEzRZ35urlk, https://medium.com/google-developer-experts/google-io-2024-summary-created-with-gemini-39b51d190fbb\n"
    # return temp_result
    search_graph = SearchGraph(
        prompt=prompt,
        config=graph_config
    )

    result = search_graph.run()

    print("Search Graph Result")
    converted_result = []
    sources = []
    convert(result, 0, converted_result, sources)
    #print(''.join(converted_result))
    #print(f"Sources: {sources}")

    print(f"Smart Scraper Multiple Graph using sources: {sources}")
    smart_scraper_multiple_graph = SmartScraperMultiGraph(
        prompt=prompt,
        config=graph_config,
        source=sources
    )
    result = smart_scraper_multiple_graph.run()

    converted_result2 = []
    sources2 = []
    convert(result, 0, converted_result2, sources2)
    #print(''.join(converted_result2))
    #print(f"Sources: {sources2}")

    res1 = ''.join(converted_result)
    res2 = ''.join(converted_result2)
    final_result = f"{res1}\n\n{res2}"
    #print(f"Final Result: {final_result}")
    return final_result

if __name__ == "__main__":
    # ************************************************
    # Create the SmartScraperGraph instance and run it
    # ************************************************


    search_graph = SearchGraph(
       prompt="Give me information about Google I/O 2024",
       config=graph_config
)

    result = search_graph.run()

    print("Search Graph Result")
    converted_result = []
    sources = []
    convert(result, 0, converted_result, sources)
    print(''.join(converted_result))
    print(f"Sources: {sources}")
    # ************************************************
    # Create the SmartScraperMultipleGraph instance and run it
    # ************************************************
    print("*" * 100)

    smart_scraper_multiple_graph = SmartScraperMultiGraph(
    prompt="Give me information about Google I/O 2024",
    config=graph_config,
    source=['https://techcrunch.com/2024/05/15/google-i-o-2024-everything-announced-so-far/',
                'https://www.youtube.com/watch?v=XEzRZ35urlk',
                'https://www.techradar.com/computing/software/google-io-2024']

    )

    result = smart_scraper_multiple_graph.run()
    print("Smart Scraper Multiple Graph Result")
    converted_result = []
    sources = []
    convert(result, 0, converted_result, sources)
    print(''.join(converted_result))
    print(f"Sources: {sources}")
    print("*" * 100)

    smart_scraper_graph1 = SmartScraperGraph(
    prompt="What were the takeaways from Google I/O 2024?",
    # also accepts a string with the already downloaded HTML code
    source="https://www.techradar.com/computing/software/google-io-2024",
    config=graph_config,
    )

    result = smart_scraper_graph1.run()
    print("Smart Scraper Graph 1 Result")
    converted_result = []
    sources = []
    convert(result, 0, converted_result, sources)
    print(''.join(converted_result))
    print(f"Sources: {sources}")
    print("*" * 100)

    smart_scraper_graph2 = SmartScraperGraph(
    prompt="Get all the details about Google I/O 2024?",
    # also accepts a string with the already downloaded HTML code
    source="https://techcrunch.com/2024/05/15/google-i-o-2024-everything-announced-so-far/",
    config=graph_config,
    )

    result = smart_scraper_graph2.run()
    print("Smart Scraper Graph 2 Result")
    converted_result = []
    sources = []
    convert(result, 0, converted_result, sources)
    print(''.join(converted_result))
    print(f"Sources: {sources}")
