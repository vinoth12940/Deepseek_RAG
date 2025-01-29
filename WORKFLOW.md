# DeepSeek RAG System Technical Workflow

```mermaid
graph TD
    subgraph Frontend["Frontend (Streamlit)"]
        A["1.1 Start"] --> B["1.2 Initialize Components"]
    end

    subgraph Database["Vector Database (Qdrant)"]
        B --> C{"2.1 Check Collection"}
        C -->|"2.2"| D["2.3 Load Collection"]
        C -->|"2.2"| E["2.3 Create Collection"]
    end

    subgraph QueryEngine["Query Engine (LlamaIndex)"]
        D --> F["3.1 Initialize Query Engine"]
        E --> F
        F --> G["3.2 Wait for User Action"]
    end

    subgraph Processing["Processing Pipeline"]
        G --> H{"4.1 User Action"}
        H -->|"4.2a"| I["4.3a Document Processing"]
        H -->|"4.2b"| J["4.3b Query Processing"]
        
        I --> K["4.4 Process Document<br/>BAAI Embeddings"]
        K --> L{"4.5 Check Duplicates"}
        L -->|"4.6a"| M["4.7a Index Document"]
        L -->|"4.6b"| N["4.7b Skip Processing"]
    end

    subgraph LLM["LLM Processing (DeepSeek)"]
        J --> O["5.1 Retrieve Context"]
        O --> P["5.2 Generate Response<br/>DeepSeek LLM"]
        P --> Q["5.3 Stream to UI"]
    end

    subgraph Monitoring["LangSmith Monitoring"]
        T["6.1 Trace Operations"]
        R["6.2 Monitor Embeddings"]
        S["6.3 Track LLM Calls"]
        
        T --> R
        R --> S
    end

    M -->|"4.8"| G
    N -->|"4.8"| G
    Q -->|"5.4"| G

    %% LangSmith Monitoring Connections
    K -.->|"Monitor"| R
    P -.->|"Monitor"| S
    B -.->|"Trace"| T
    F -.->|"Trace"| T
    O -.->|"Trace"| T

    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px
    classDef process fill:#bbdefb,stroke:#333,stroke-width:2px
    classDef special fill:#ffecb3,stroke:#333,stroke-width:2px
    classDef monitor fill:#e8f5e9,stroke:#333,stroke-width:2px
    
    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q process
    class Frontend,Database,QueryEngine,Processing,LLM special
    class T,R,S monitor
```

Flow Sequence:
1. Frontend (Streamlit)
   - 1.1 Application Start
   - 1.2 Initialize Components

2. Vector Database (Qdrant)
   - 2.1 Check Collection
   - 2.2 Determine Collection Status
   - 2.3 Load or Create Collection

3. Query Engine (LlamaIndex)
   - 3.1 Initialize Query Engine
   - 3.2 Wait for User Action

4. Processing Pipeline
   - 4.1 User Action Decision
   - 4.2a/b Branch to Document or Query Processing
   - 4.3a Document Processing Path
   - 4.3b Query Processing Path
   - 4.4 Process Document with BAAI Embeddings
   - 4.5 Check for Duplicates
   - 4.6a/b New or Duplicate Document
   - 4.7a Index New Document
   - 4.7b Skip Duplicate Document
   - 4.8 Return to Wait State

5. LLM Processing (DeepSeek)
   - 5.1 Retrieve Relevant Context
   - 5.2 Generate Response using DeepSeek LLM
   - 5.3 Stream to UI
   - 5.4 Return to Wait State

6. LangSmith Monitoring
   - 6.1 Trace All Operations
   - 6.2 Monitor Embedding Generation
   - 6.3 Track LLM Interactions