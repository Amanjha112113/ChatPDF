// services/geminiService.ts

const API_BASE_URL = process.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface ChatSession {
    sendMessage: (message: string) => Promise<string>;
}

// 1. Update uploadPdf to RETURN the doc_id
export async function uploadPdf(text: string): Promise<string> {
    const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to upload PDF');
    }
    
    // CAPTURE and RETURN the response data
    const data = await response.json();
    if (!data.doc_id) {
        throw new Error("Server did not return a doc_id.");
    }
    return data.doc_id;
}

// 2. Update queryPdf to ACCEPT and SEND the doc_id
export async function queryPdf(query: string, doc_id: string): Promise<string> {
    const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        // SEND BOTH query and doc_id
        body: JSON.stringify({ query, doc_id }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to query PDF');
    }

    const data = await response.json();
    return data.answer;
}

// 3. Update createChatSession to "hold" the doc_id in a closure
export async function createChatSession(doc_id: string): Promise<ChatSession> {
    // This function's only job is to create an object
    // that "remembers" the doc_id for all its method calls.
    return {
        sendMessage: async (message: string) => {
            // Pass the "remembered" doc_id to queryPdf
            return await queryPdf(message, doc_id);
        }
    };
}