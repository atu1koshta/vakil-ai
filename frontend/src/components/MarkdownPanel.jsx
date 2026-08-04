import Editor from '@monaco-editor/react';

export default function MarkdownPanel({ markdown }) {
  return (
    <Editor
      height="100%"
      defaultLanguage="markdown"
      value={markdown}
      options={{
        readOnly: true,
        wordWrap: 'on',
        minimap: { enabled: false },
        fontSize: 13,
        lineNumbers: 'on',
        scrollBeyondLastLine: false,
      }}
    />
  );
}
