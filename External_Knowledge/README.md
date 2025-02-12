# Query Enhancement, Web Search, Scraping, and Summarization Tool

## Overview

This Python script allows users to:
1. Enhance their queries using Qwen AI.
2. Generate multiple related queries from a single input query.
3. Perform web searches using DuckDuckGo for each generated query.
4. Scrape content from the top results of these searches.
5. Summarize the scraped data concisely while citing sources.

The tool is designed to automate search and summarization tasks by leveraging AI-powered query enhancement (via Qwen) and web scraping techniques.

---

## Requirements

The following Python libraries are required:
- **`requests`**: For making HTTP requests to fetch webpage content.
- **`beautifulsoup4`**: For parsing HTML content during web scraping.
- **`duckduckgo-search`**: To perform web searches via DuckDuckGo's API wrapper.
- **`ninept`**: To integrate with Qwen AI for query enhancement and summarization.

See [`Requirements_ExternalKnowledge.txt`](./requirements.txt) for exact versions of these dependencies.

---

##Install Dependencies 
pip install -r Requirements_ExternalKnowledge.txt


## Example Workflow

Here’s an example flow that demonstrates how this tool works:

### Step 1: Run the Script
Run this command from your terminal after setting up dependencies:

```bash
python3 external_knowledge.py
```
### Step 2: Input you Query
munich rent index 2023  (example)


### Step 3: View Generated Queries & Search Results

Qwen AI will generate multiple related queries based on your input:
Searching the web...

Searching with query: 1. Munich housing price index

Searching with query: 2. Rent trends in Munich

Searching with query: 3. Cost of living: Munich rental market

Searching with query: 4. Munich apartment rental costs

Searching with query: 5. Average rent in Munich by neighborhood

Searching with query: 6. Comparison of Munich rent to other German cities

Searching with query: 7. Munich rent increase statistics

Searching with query: 8. Rental index for Munich apartments and houses

Searching with query: 9. Munich real estate market analysis:租金版块

Searching with query: 10. Impact of COVID-19 on Munich rent prices

Searching with query: 11. Long-term vs short-term rental rates in Munich

Searching with query: 12. Student housing rent in Munich

Searching with query: 13. Luxury rent index in Munich

Searching with query: 14. Munich rental market forecast

Searching with query: 15. Budget-friendly areas for renting in Munich

Searching with query: 16. Historical rent data for Munich

Searching with query: 17. Munich property rental yield comparison

Searching with query: 18. Rent vs. buy analysis for Munich properties

Searching with query: 19. Munich rental market trends 2022

Searching with query: 20. Furnished rental options in Munich and their prices.

### Step 4: Scraping & Summarization Output 

Finally, after scraping content from top search results, Qwen AI summarizes them into concise paragraphs with citations(Currently the websites count is set to 5, but we can eaily increase or decrease the website count):

Scraping content from top search results...
Error scraping URL (https://www.globalpropertyguide.com/europe/germany/price-history): 403 Client Error: Forbidden for url: https://www.globalpropertyguide.com/europe/germany/price-history

Generating summary...


Summary of Results:
Construction and real estate prices in Germany, specifically in Munich, have experienced significant growth over the past few years. As of December 20, 2024, the house price index showed a decrease of 0.7% compared to the third quarter of 2023, but a 0.3% increase from the previous quarter. Munich stands out as having the highest property price level in the country, driven by high demand due to its cultural, economic, and political significance.

In Munich, average rental prices are the highest in Germany, with tenants paying 18.33 euros per square meter for existing properties and 20.62 euros for new buildings in Q2 2020. Asking prices for condominiums have also risen, with new constructions fetching an average of 10,431 euros per square meter, a 10.5% increase from 2019.

Single-family home prices have seen a substantial increase of 26% since 2015, reaching an average asking price of 1.512.456 euros in 2020. Property prices vary across Munich's districts, with prime locations like the old town, Altschwabing, and Altbogenhausen having prices ranging from 10,000 to 25,000 euros per square meter for condominiums.

Despite the COVID-19 pandemic, experts predict that Munich's property prices will continue to rise due to persistent high demand. The market for residential and commercial buildings remains strong, with a high demand overhang in Munich. However, the market might experience adjustments due to tightened purchasing conditions in conservation areas and increased competition in other zones.

For more detailed insights and market data, consult the sources provided, including the German Federal Statistical Office (Destatis) and Engel & Völkers, which offer in-depth analyses and forecasts for the Munich and national real estate markets.
