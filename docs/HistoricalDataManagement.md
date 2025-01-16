mermaid'''
flowchart TD
    subgraph HistoricalDataManager ["HistoricalDataManager Class"]
        init["__init__(base_path: str = 'data', debug: bool = False)"]
        getPrice["get_price_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame"]
        getNews["get_news_data(symbol: str, start_date: datetime, end_date: datetime) -> List[dict]"]
        clearCache["clear_cache(data_type: Optional[str] = None)"]
    end

    subgraph Dependencies
        priceCache["_price_cache: Dict[str, pd.DataFrame]"]
        newsCache["_news_cache: Dict[str, List[dict]]"]
        fundamentalsCache["_fundamentals_cache: Dict[str, dict]"]
        basePath["base_path: str"]
        pricePath["price_path: Path"]
        newsPath["news_path: Path"]
        fundamentalsPath["fundamentals_path: Path"]
    end

    init --> basePath & pricePath & newsPath & fundamentalsPath & priceCache & newsCache & fundamentalsCache
    getPrice --> priceCache
    getNews --> newsCache
    clearCache --> priceCache & newsCache & fundamentalsCache

    classDef function fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000;
    classDef dependency fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000;

    class init,getPrice,getNews,clearCache function;
    class priceCache,newsCache,fundamentalsCache,basePath,pricePath,newsPath,fundamentalsPath dependency;
    '''
