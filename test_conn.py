import MetaTrader5 as mt5
from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

# კავშირის შემოწმება
if mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
    acc = mt5.account_info()
    print("=" * 40)
    print("✅ წარმატებით დაუკავშირდა!")
    print(f"📌 ანგარიში: {acc.login}")
    print(f"💰 ბალანსი: {acc.balance} {acc.currency}")
    print(f"🏦 სერვერი: {acc.server}")
    print(f"🏢 ბროკერი: {acc.company}")
    print("=" * 40)
    mt5.shutdown()
else:
    print("❌ კავშირი ვერ დამყარდა!")
    print(f"შეცდომის კოდი: {mt5.last_error()}")