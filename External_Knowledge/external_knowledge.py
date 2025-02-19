# Required Libraries
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS  # For DuckDuckGo Search
from ninept import qwen            # For Query Enhancement &; Summarization

# Function: Generate Multiple Queries Using Qwen
def generate_queries_with_qwen(query):
    try:
        prompt = f"Generate multiple variations of this search query that cover related topics:\n\n{query}"
        generated_queries = qwen(content=prompt, role="You are a helpful assistant who generates diverse search queries.")
        
        # Split by lines or commas to handle multiple outputs from Qwen
        return [q.strip() for q in generated_queries.split("\n") if q.strip()]
    
    except Exception as e:
        print(f"Error generating queries with Qwen: {e}")
        return [query]  # Fallback to original query if generation fails

# Function: Perform Web Search Using DuckDuckGo (for Multiple Queries)
def perform_duckduckgo_search_multiple(queries):
    try:
        ddgs = DDGS()
        aggregated_results = []
        
        for query in queries:
            print(f"\nSearching with query: {query}")
            
            results = []
            for result in ddgs.text(query, max_results=5):  # Fetch top 5 results per query
                results.append((result['href'], result['title']))
            
            aggregated_results.extend(results)
        
        # Remove duplicate URLs by converting to a dictionary (preserves order)
        unique_results = list({url: title for url, title in aggregated_results}.items())
        
        return unique_results
    
    except Exception as e:
        print(f"Error performing DuckDuckGo search: {e}")
        return []

# Function: Scrape Web Content from URLs (Improved with Headers)
def scrape_web_content(urls):
    contents = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    for url, title in urls[:5]:  # Process up to top 5 URLs only
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise error if request fails
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract main content (simplified extraction logic)
            paragraphs = soup.find_all('p')
            text_content = " ".join([p.get_text() for p in paragraphs])
            
            contents.append((title, url, text_content))
        
        except Exception as e:
            print(f"Error scraping URL ({url}): {e}")
    
    return contents

# Function: Summarize Content Using Qwen
def summarize_with_qwen(contents):
    try:
        combined_text = "\n\n".join([f"Source Title: {title}\nSource URL: {url}\n{text}" 
                                     for title, url, text in contents])
        
        prompt = f"Please summarize the following information concisely while citing sources:\n\n{combined_text}"
        
        summary = qwen(content=prompt.strip(), role="You are a helpful assistant who summarizes information.")
        
        return summary.strip()
    
    except Exception as e:
        print(f"Error summarizing content with Qwen: {e}")
        return "An error occurred during summarization."

# Main Program Flow
if __name__ == "__main__":
    print("Welcome! Please enter your query:")
    
    user_query = input("> ").strip()  # Step 1: Get User Input
    
    if not user_query:
        print("Query cannot be empty! Exiting.")
    
    else:
        print("\nGenerating multiple related queries...")
        
        generated_queries = generate_queries_with_qwen(user_query)  # Step 2: Generate Multiple Queries
        
        print(f"\nGenerated Queries:\n{generated_queries}")
        
        print("\nSearching the web...")
        
        search_results = perform_duckduckgo_search_multiple(generated_queries)  # Step 3 &; Step 4
        
        if not search_results or len(search_results) == 0:
            print("No relevant results found. Exiting.")
        
        else:
            print("\nScraping content from top search results...")
            
            scraped_contents = scrape_web_content(search_results)  # Step 5
            
            if not scraped_contents or len(scraped_contents) == 0:
                print("Failed to scrape any meaningful content. Exiting.")
            
            else:
                print("\nGenerating summary...")
                
                final_summary = summarize_with_qwen(scraped_contents)  # Step 6
                
                print("\nSummary of Results:")
                print(final_summary)



