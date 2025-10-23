import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ChatMessage } from './ChatMessage';
import { SendIcon } from './icons';
import type { ChatMessage as ChatMessageType } from '../types';

import type { ChatSession } from '../services/geminiService';

interface ChatViewProps {
  chatSession: ChatSession;
}

export const ChatView: React.FC<ChatViewProps> = ({ chatSession }) => {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [userInput, setUserInput] = useState<string>('');
  const [isReplying, setIsReplying] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, isReplying]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userInput.trim() || isReplying) return;

    const userMessage: ChatMessageType = { role: 'user', text: userInput };
    setMessages(prev => [...prev, userMessage]);
    setUserInput('');
    setIsReplying(true);
    setError(null);

    try {
      const response = await chatSession.sendMessage(userInput);
      const modelMessage: ChatMessageType = { role: 'model', text: response };
      setMessages(prev => [...prev, modelMessage]);
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred.';
      console.error(e);
      setError(`Failed to get response: ${errorMessage}`);
    } finally {
      setIsReplying(false);
    }
  };

  return (
    <div className="flex flex-col h-full w-full max-w-4xl mx-auto bg-brand-surface rounded-xl shadow-2xl border border-brand-border">
      <div ref={chatContainerRef} className="flex-grow p-6 overflow-y-auto space-y-6">
        {messages.length === 0 && (
          <div className="text-center text-brand-text-secondary">
            <p>Chat session started. Ask me anything about the document!</p>
          </div>
        )}
        {messages.map((msg, index) => (
          <ChatMessage key={index} role={msg.role} text={msg.text} />
        ))}
        {isReplying && <ChatMessage role="model" text="" isLoading={true} />}
        {error && <div className="text-red-400 bg-red-900/50 p-3 rounded-lg">{error}</div>}
      </div>
      <div className="p-4 border-t border-brand-border">
        <form onSubmit={handleSubmit} className="flex items-center gap-4">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Ask a question about your document..."
            className="flex-grow bg-brand-bg border border-brand-border rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-brand-primary w-full"
            disabled={isReplying}
          />
          <button
            type="submit"
            disabled={isReplying || !userInput.trim()}
            className="bg-brand-primary text-white p-3 rounded-lg disabled:bg-brand-border disabled:cursor-not-allowed hover:bg-brand-primary-hover transition-colors duration-200"
          >
            <SendIcon className="w-6 h-6" />
          </button>
        </form>
      </div>
    </div>
  );
};
