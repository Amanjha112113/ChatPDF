import React from 'react';
import { marked } from 'marked';
import { UserIcon, ModelIcon } from './icons';

interface ChatMessageProps {
  role: 'user' | 'model';
  text: string;
  isLoading?: boolean;
}

const LoadingIndicator = () => (
  <div className="flex items-center space-x-2">
    <div className="w-2 h-2 bg-brand-text-secondary rounded-full animate-pulse [animation-delay:-0.3s]"></div>
    <div className="w-2 h-2 bg-brand-text-secondary rounded-full animate-pulse [animation-delay:-0.15s]"></div>
    <div className="w-2 h-2 bg-brand-text-secondary rounded-full animate-pulse"></div>
  </div>
);


export const ChatMessage: React.FC<ChatMessageProps> = ({ role, text, isLoading }) => {
  const isUser = role === 'user';
  
  const createMarkup = (markdownText: string) => {
    // We trust the output from the Gemini API.
    // For user-generated content, you would want to sanitize the HTML.
    const rawMarkup = marked.parse(markdownText, { gfm: true, breaks: true }) as string;
    return { __html: rawMarkup };
  };

  return (
    <div className={`flex items-start gap-4 ${isUser ? 'justify-end' : ''}`}>
      {!isUser && (
        <div className="w-8 h-8 flex-shrink-0 bg-brand-border rounded-full flex items-center justify-center">
          <ModelIcon className="w-5 h-5 text-brand-primary" />
        </div>
      )}
      <div className={`max-w-xl p-4 rounded-2xl ${isUser ? 'bg-brand-primary text-white rounded-br-none' : 'bg-brand-bg text-brand-text rounded-bl-none'}`}>
        {isLoading ? (
          <LoadingIndicator />
        ) : (
          isUser ? (
             <p className="whitespace-pre-wrap">{text}</p>
          ) : (
            <div 
              className="prose prose-invert max-w-none"
              dangerouslySetInnerHTML={createMarkup(text)} 
            />
          )
        )}
      </div>
       {isUser && (
        <div className="w-8 h-8 flex-shrink-0 bg-brand-border rounded-full flex items-center justify-center">
          <UserIcon className="w-5 h-5 text-brand-text-secondary" />
        </div>
      )}
    </div>
  );
};