# 🚀 Setup WhatsApp Integration - Fase 1

## 📋 Pré-requisitos

1. ✅ Ambiente conda `bts_final_project` ativo
2. ✅ Twilio instalado (`pip install twilio`)
3. ⏳ Conta Twilio (gratuita)
4. ⏳ ngrok instalado

## 🔧 Configuração Passo a Passo

### 1. Criar Conta Twilio (Gratuita)
1. Acesse: https://console.twilio.com/
2. Crie uma conta gratuita
3. Anote seu **Account SID** e **Auth Token**

### 2. Configurar Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto:

```bash
# .env
TWILIO_ACCOUNT_SID=seu_account_sid_aqui
TWILIO_AUTH_TOKEN=seu_auth_token_aqui
WEBHOOK_URL=http://localhost:8001
```

### 3. Instalar ngrok (se não tiver)
```bash
# macOS
brew install ngrok

# Ou baixe de: https://ngrok.com/download
```

### 4. Ativar WhatsApp Sandbox no Twilio
1. No console Twilio, vá em: **Messaging** → **Try it out** → **Send a WhatsApp message**
2. Siga as instruções para conectar seu WhatsApp ao sandbox
3. Anote o número do sandbox (ex: `+1 415 523 8886`)

## 🚀 Executar Fase 1

### 1. Ativar ambiente e rodar webhook
```bash
conda activate bts_final_project
cd app
python -m whatsapp_integration.run
```

### 2. Em outro terminal, expor com ngrok
```bash
ngrok http 8001
```

### 3. Configurar webhook no Twilio
1. Copie a URL do ngrok (ex: `https://abc123.ngrok.io`)
2. No console Twilio, vá em WhatsApp Sandbox Settings
3. Configure webhook URL: `https://abc123.ngrok.io/whatsapp`

## ✅ Testar Fase 1

1. Envie uma mensagem de texto para o número do sandbox
2. Deve receber um eco da mensagem
3. Verifique os logs no terminal

### Mensagens de Teste:
- `"oi"` → Resposta personalizada
- `"hello"` → Resposta personalizada  
- Qualquer texto → Echo com timestamp

## 🔍 Verificar Status

Acesse: http://localhost:8001/ para ver status da configuração

## 🐛 Troubleshooting

### Erro: "Twilio credentials not configured"
- Verifique se o arquivo `.env` existe na raiz do projeto
- Confirme se as variáveis estão corretas

### Erro: "Connection refused"
- Verifique se ngrok está rodando
- Confirme se a URL do webhook no Twilio está correta

### Não recebe mensagens
- Verifique se seguiu o processo de ativação do sandbox
- Confirme se enviou a mensagem de ativação para o Twilio

## 📝 Próximos Passos

Após a Fase 1 funcionar:
- ✅ Fase 1: Echo de mensagens de texto
- ⏳ Fase 2: Recebimento de áudio com resposta fixa
- ⏳ Fase 3: Integração com API de análise
- ⏳ Fase 4: Testes finais e documentação 