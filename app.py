import streamlit as st
st.set_page_config(
  page_title="predic_USDJPY_H1_ScikitLearn app",
  page_icon="🚁",
)
st.title("USDJPY1時間足予測(Scikit-Learn)アプリ")
st.markdown('## 概要及び注意事項')
st.write("当アプリでは、USDJPYの1時間足を直近のデータ(yahoo finance)に基づき陽線か、陰線かをScikit-Learnを使用して予測します。ただし本結果により投資にいかなる損失が生じても、当アプリでは責任を取りません。あくまで参考程度にご利用ください。")
st.write('なお時刻はUTC(日本時間マイナス9時間)の表示となります。')
try:
    if st.button('予測開始'):
        comment = st.empty()
        comment.write('予測を開始しました')

        import time
        t1 = time.time()
        import numpy as np
        import csv
        from sklearn import svm
        import math
        import pandas as pd
        import yfinance as yf

        # 外為データ取得
        tks  = 'USDJPY=X'
        data = yf.download(tickers  = tks ,          # 通貨ペア
                        period   = '1mo',          # データ取得期間 15m,1d,1mo,3mo,1y,10y,20y,30y  1996年10月30日からデータがある。
                        interval = '1h',         # データ表示間隔 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                        )

        #最後の日時を取り出す。
        lastdatetime = data.index[-1]

        #Close価格のみを取り出す。
        data_close = data['Close']

        #対数表示に変換する
        ln_fx_price = []
        for line in data_close:
            ln_fx_price.append(math.log(line))
        count_s = len(ln_fx_price)

        # 為替の上昇率を算出、おおよそ-1.0-1.0の範囲に収まるように調整
        modified_data = []
        for i in range(1, count_s):
            modified_data.append(float(ln_fx_price[i] - ln_fx_price[i-1])*1000)
        count_m = len(modified_data)

        # 前日までの4連続の上昇率のデータ
        successive_data = []
        # 正解値 価格上昇: 1 価格下落: 0
        answers = []
        for i in range(4, count_m):
            successive_data.append([modified_data[i-4], modified_data[i-3], modified_data[i-2], modified_data[i-1]])
            if modified_data[i] > 0:
                answers.append(1)
            else:
                answers.append(0)
        # print (successive_data)
        # print (answers)

        # データ数
        n = len(successive_data)
        # print (n)
        m = len(answers)
        # print (m)

        # 線形サポートベクターマシーン
        clf = svm.LinearSVC()
        # サポートベクターマシーンによる訓練 （データの75%を訓練に使用）
        clf.fit(successive_data[:int(n*750/1000)], answers[:int(n*750/1000)])

        # テスト用データ
        # 正解
        expected = answers[int(-n*250/1000):]
        # 予測
        predicted = clf.predict(successive_data[int(-n*250/1000):])

        st.write(f'{lastdatetime}の次の1時間足の予測')
        # 末尾の10個を比較
        #print ('正解:' + str(expected[-10:]))
        #print ('予測:' + str(list(predicted[-10:])))

        # 正解率の計算
        correct = 0.0
        wrong = 0.0
        for i in range(int(n*250/1000)):
            if expected[i] == predicted[i]:
                correct += 1
            else:
                wrong += 1
                
        #print('正解数： ' + str(int(correct)))
        #print('不正解数： ' + str(int(wrong)))

        successive_data.append([modified_data[count_m-4], modified_data[count_m-3], modified_data[count_m-2], modified_data[count_m-1]])
        predicted = clf.predict(successive_data[-1:])
        #print ('次の1時間足の予測:' + str(list(predicted)) + ' 1:陽線,　0:陰線')
        if str(list(predicted)) == str([1]):
            st.write('「陽線」でしょう。')
        else:
            st.write('「陰線」でしょう。')
        st.write("正解率: " + str(round(correct / (correct+wrong) * 100,  2)) + "%")   
        t2 = time.time()
        elapsed_time = t2- t1
        elapsed_time = round(elapsed_time, 2)
        st.write('プログラム処理時間： ' + str(elapsed_time) + '秒')
        comment.write('完了しました！')
except:
    st.error('エラーが発生しました。しばらくしてから、再度実行してください。')
