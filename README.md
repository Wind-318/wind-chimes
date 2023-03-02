# Wind chimes

## OpenAI Chat API Implementation
This package provides an implementation of the OpenAI Chat API for GPT-3.5, allowing you to interact with the API using Go.

### Quick Start
- Import the openai package:
```Go
import "github.com/Wind-318/wind-chimes/openai"
```

- Create a new Chat object:
```Go
chat := &openai.Chat{
	Data: make(map[string]interface{}),
}
```

- Set the authorization key for the API:
```Go
chat.SetAuthorizationKey("YOUR_API_KEY")
```

- Add messages to the chat:
```Go
chat.AddMessage("user", "Hello")
```

- Set any additional parameters for the chat(optional):
```Go
chat.SetTemperature(0.7)
chat.SetTopP(0.9)
chat.SetN(1)
```

- Send the chat request to the API:
```Go
resp, err := chat.Send()
if err != nil {
    // Handle error
}

for _, choice := range resp.Choices {
    fmt.Println(choice.Message.Content)
}
```