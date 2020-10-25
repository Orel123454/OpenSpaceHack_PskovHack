import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import org.telegram.telegrambots.ApiContextInitializer
import org.telegram.telegrambots.bots.TelegramLongPollingBot
import org.telegram.telegrambots.meta.TelegramBotsApi
import org.telegram.telegrambots.meta.api.methods.send.SendMessage
import org.telegram.telegrambots.meta.api.objects.Update
import org.telegram.telegrambots.meta.api.objects.replykeyboard.ReplyKeyboardMarkup
import org.telegram.telegrambots.meta.api.objects.replykeyboard.ReplyKeyboardRemove
import org.telegram.telegrambots.meta.api.objects.replykeyboard.buttons.KeyboardButton
import org.telegram.telegrambots.meta.api.objects.replykeyboard.buttons.KeyboardRow
import java.io.File
import java.net.URL


fun main() {

    ApiContextInitializer.init()
    TelegramBotsApi().registerBot(Bot())

}

class Bot : TelegramLongPollingBot() {
    private var parser: Parser = Parser.default()
    private var stringBuilder: StringBuilder = StringBuilder(File("./package.json").readText())
    private var json: JsonObject = parser.parse(stringBuilder) as JsonObject

    override fun getBotToken(): String {
        return "1357410487:AAG3jRz52vIIaARRdwr-4IShYqHCfTI1ZC8"
    }

    private fun sendGet(param: String): String? {
        val url = URL("127.0.0.1/?&rgu=$param")
        val answer: JsonObject = parser.parse(StringBuilder(url.readText())) as JsonObject
        return answer.string("title")
    }

    override fun onUpdateReceived(p0: Update?) {

        when (p0!!.message.text) {
            "/start" -> {
            execute(SendMessage()
                    .setChatId(p0.message.chatId)
                    .setText(json.string("greeting"))

            )
        }
        "Вернуться", "Ответ на \"Не мой вопрос\"" ->

            execute(SendMessage()
                    .setReplyMarkup(ReplyKeyboardRemove())
                    .setChatId(p0.message.chatId)
                    .setText(json.string("comeback"))
            )
        "Я полезный?" -> {
            val keyboard = ReplyKeyboardMarkup()

            keyboard.keyboard = listOf(
                    KeyboardRow().apply{
                        add(KeyboardButton("Да"))
                        add(KeyboardButton("Нет"))
                        add(KeyboardButton("Не знаю"))
                    },
            )
            keyboard.resizeKeyboard = true

            execute(SendMessage()
                    .setReplyMarkup(keyboard)
                    .setChatId(p0.message.chatId)
                    .setText(json.string("ynidn"))
            )
        }
        else ->
            try {
                val keyboard = ReplyKeyboardMarkup()

                keyboard.keyboard = listOf(
                        KeyboardRow().apply{
                            add(KeyboardButton("Вернуться"))
                            add(KeyboardButton("Ответ на \"Не мой вопрос\""))
                        },
                        KeyboardRow().apply{
                            add(KeyboardButton("Я полезный?"))
                        }
                )

                keyboard.resizeKeyboard = true

                execute(SendMessage()
                        .setReplyMarkup(keyboard)
                        .setChatId(p0.message.chatId)
                        .setText((sendGet(p0.message.text))))

            } catch (e: Exception) {
                execute(SendMessage()
                        .setChatId(p0.message.chatId)
                        .setText(json.string("error")))
            }
        }
    }

    override fun getBotUsername(): String {
        return "test_osph_bot"
    }
}