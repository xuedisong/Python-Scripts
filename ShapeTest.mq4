//+------------------------------------------------------------------+
//|                                                    ShapeTest.mq4 |
//|                        Copyright 2017, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart(){
   
   ObjectsDeleteAll();
//---定义乌云盖顶,行情要下跌。
//刻画符合“乌云盖顶”形态的连续两个蜡烛实体
//收盘价小于开盘价
//前一根，收盘价大于开盘价
//开盘价大于前一天的收盘价
//收盘价小于前一天开盘及收盘的平均
//收盘价大于前一天的开盘价

   const int num=1000;//作用的bar的数目
   int Cloud[1000];//静态数组
   for(int i=0;i<num;i++){
      if(Close[i]<Open[i]&&Close[i+1]>Open[i+1]&&Open[i]>Close[i+1]&&Close[i]<(Close[i+1]+Open[i+1])/2&&Close[i]>Open[i+1])
         Cloud[i]=1;
   }
   
//---定义上升趋势   
//定义前期上升趋势
   int Trend[998];//?????
   for(int i=0;i<num-2;i++){
      if(Close[i+1]>Close[i+2]&&Close[i+2]>Close[i+3]){
         Trend[i]=1;
      }
   }
   
//寻找“乌云盖顶”形态
   int darkCloud[998];//?????动态数组
   int count=0;
   for(int i=0;i<num-2;i++){
      darkCloud[i]=Trend[i]+Cloud[i];
      if(darkCloud[i]==2){
         ObjectCreate("testdraw" + IntegerToString(i),OBJ_ARROW_SELL, 0, Time[i], Close[i]);        
         count+=1;
      }
   }
   printf("反转信号有"+count+"个");
}

//+------------------------------------------------------------------+
