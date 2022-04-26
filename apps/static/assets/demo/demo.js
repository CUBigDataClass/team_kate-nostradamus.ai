type = ['primary', 'info', 'success', 'warning', 'danger'];

demo = {
  initPickColor: function() {
    $('.pick-class-label').click(function() {
      var new_class = $(this).attr('new-class');
      var old_class = $('#display-buttons').attr('data-class');
      var display_div = $('#display-buttons');
      if (display_div.length) {
        var display_buttons = display_div.find('.btn');
        display_buttons.removeClass(old_class);
        display_buttons.addClass(new_class);
        display_div.attr('data-class', new_class);
      }
    });
  },

  initDocChart: function() {
    chartColor = "#FFFFFF";

    // General configuration for the charts with Line gradientStroke
    gradientChartOptionsConfiguration = {
      maintainAspectRatio: false,
      legend: {
        display: false
      },
      tooltips: {
        bodySpacing: 4,
        mode: "nearest",
        intersect: 0,
        position: "nearest",
        xPadding: 10,
        yPadding: 10,
        caretPadding: 10
      },
      responsive: true,
      scales: {
        yAxes: [{
          display: 0,
          gridLines: 0,
          ticks: {
            display: false
          },
          gridLines: {
            zeroLineColor: "transparent",
            drawTicks: false,
            display: false,
            drawBorder: false
          }
        }],
        xAxes: [{
          display: 0,
          gridLines: 0,
          ticks: {
            display: false
          },
          gridLines: {
            zeroLineColor: "transparent",
            drawTicks: false,
            display: false,
            drawBorder: false
          }
        }]
      },
      layout: {
        padding: {
          left: 0,
          right: 0,
          top: 15,
          bottom: 15
        }
      }
    };

    ctx = document.getElementById('lineChartExample').getContext("2d");

    gradientStroke = ctx.createLinearGradient(500, 0, 100, 0);
    gradientStroke.addColorStop(0, '#80b6f4');
    gradientStroke.addColorStop(1, chartColor);

    gradientFill = ctx.createLinearGradient(0, 170, 0, 50);
    gradientFill.addColorStop(0, "rgba(128, 182, 244, 0)");
    gradientFill.addColorStop(1, "rgba(249, 99, 59, 0.40)");

    myChart = new Chart(ctx, {
      type: 'line',
      responsive: true,
      data: {
        labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        datasets: [{
          label: "Active Users",
          borderColor: "#f96332",
          pointBorderColor: "#FFF",
          pointBackgroundColor: "#f96332",
          pointBorderWidth: 2,
          pointHoverRadius: 4,
          pointHoverBorderWidth: 1,
          pointRadius: 4,
          fill: true,
          backgroundColor: gradientFill,
          borderWidth: 2,
          data: [542, 480, 430, 550, 530, 453, 380, 434, 568, 610, 700, 630]
        }]
      },
      options: gradientChartOptionsConfiguration
    });
  },

  initDashboardPageCharts: function() {

    gradientChartOptionsConfigurationWithTooltipBlue = {
      maintainAspectRatio: false,
      legend: {
        display: false
      },

      tooltips: {
        backgroundColor: '#f5f5f5',
        titleFontColor: '#333',
        bodyFontColor: '#666',
        bodySpacing: 4,
        xPadding: 12,
        mode: "nearest",
        intersect: 0,
        position: "nearest"
      },
      responsive: true ,
      scales: {
        yAxes: [{
          barPercentage: 1.6,
          gridLines: {
            drawBorder: false,
            color: 'rgba(29,140,248,0.0)',
            zeroLineColor: "transparent",
          },
          ticks: {
            suggestedMin: 50000
            ,
            suggestedMax: 125,
            padding: 20,
            fontColor: "#2380f7"
          }
        }],

        xAxes: [{
          barPercentage: 1.6,
          gridLines: {
            drawBorder: false,
            color: 'rgba(29,140,248,0.1)',
            zeroLineColor: "transparent",
          },
          ticks: {
            padding: 20,
            fontColor: "#2380f7"
          }
        }]
      }
    };

    gradientChartOptionsConfigurationWithTooltipPurple = {
      maintainAspectRatio: false,
      legend: {
        display: false
      },

      tooltips: {
        backgroundColor: '#f5f5f5',
        titleFontColor: '#333',
        bodyFontColor: '#666',
        bodySpacing: 4,
        xPadding: 12,
        mode: "nearest",
        intersect: 0,
        position: "nearest"
      },
      responsive: true,
      scales: {
        yAxes: [{
          barPercentage: 1.6,
          gridLines: {
            drawBorder: false,
            color: 'rgba(29,140,248,0.0)',
            zeroLineColor: "transparent",
          },
          ticks: {
            suggestedMin: 50000,
            suggestedMax: 125,
            padding: 20,
            fontColor: "#9a9a9a"
          }
        }],

        xAxes: [{
          barPercentage: 1.6,
          gridLines: {
            drawBorder: false,
            color: 'rgba(225,78,202,0.1)',
            zeroLineColor: "transparent",
          },
          ticks: {
            padding: 20,
            fontColor: "#9a9a9a"
          }
        }]
      }
    };

    gradientChartOptionsConfigurationWithTooltipOrange = {
      maintainAspectRatio: false,
      legend: {
        display: false
      },

      tooltips: {
        backgroundColor: '#f5f5f5',
        titleFontColor: '#333',
        bodyFontColor: '#666',
        bodySpacing: 4,
        xPadding: 12,
        mode: "nearest",
        intersect: 0,
        position: "nearest"
      },
      responsive: true,
      scales: {
        yAxes: [{
          barPercentage: 1.6,
          gridLines: {
            drawBorder: false,
            color: 'rgba(29,140,248,0.0)',
            zeroLineColor: "transparent",
          },
          ticks: {
            suggestedMin: 50000,
            suggestedMax: 110,
            padding: 20,
            fontColor: "#ff8a76"
          }
        }],

        xAxes: [{
          barPercentage: 1.6,
          gridLines: {
            drawBorder: false,
            color: 'rgba(220,53,69,0.1)',
            zeroLineColor: "transparent",
          },
          ticks: {
            padding: 20,
            fontColor: "#ff8a76"
          }
        }]
      }
    };

    gradientChartOptionsConfigurationWithTooltipGreen = {
      maintainAspectRatio: false,
      legend: {
        display: false
      },

      tooltips: {
        backgroundColor: '#f5f5f5',
        titleFontColor: '#333',
        bodyFontColor: '#666',
        bodySpacing: 4,
        xPadding: 12,
        mode: "nearest",
        intersect: 0,
        position: "nearest"
      },
      responsive: true,
      scales: {
        yAxes: [{
          barPercentage: 1.6,
          gridLines: {
            drawBorder: false,
            color: 'rgba(29,140,248,0.0)',
            zeroLineColor: "transparent",
          },
          ticks: {
            suggestedMin: 50000,
            suggestedMax: 125,
            padding: 20,
            fontColor: "#9e9e9e"
          }
        }],

        xAxes: [{
          barPercentage: 1.6,
          gridLines: {
            drawBorder: false,
            color: 'rgba(0,242,195,0.1)',
            zeroLineColor: "transparent",
          },
          ticks: {
            padding: 20,
            fontColor: "#9e9e9e"
          }
        }]
      }
    };

   
  
 

    // var ctx = document.getElementById("chartBig1").getContext("2d");

    // var gradientStroke = ctx.createLinearGradient(0, 230, 0, 1000);

    // gradientStroke.addColorStop(1, 'rgba(72,72,176,0.2)');
    // gradientStroke.addColorStop(0.2, 'rgba(72,72,176,0.0)');
    // gradientStroke.addColorStop(0, 'rgba(119,52,169,0)'); //purple colors

    

   


    // var myChart = new Chart(ctx, {
    //   type: 'line',
    //   data: data,
    //   options: gradientChartOptionsConfigurationWithTooltipPurple
    // });


    // var ctxGreen = document.getElementById("chartBig1").getContext("2d");

    // var gradientStroke = ctx.createLinearGradient(0, 230, 0, 50);

    // gradientStroke.addColorStop(1, 'rgba(66,134,121,0.15)');
    // gradientStroke.addColorStop(0.4, 'rgba(66,134,121,0.0)'); //green colors
    // gradientStroke.addColorStop(0, 'rgba(66,134,121,0)'); //green colors

    // var data = {
    //   labels: ['JUL', 'AUG', 'SEP', 'OCT', 'NOV'],
    //   datasets: [{
    //     label: "My First dataset",
    //     fill: true,
    //     backgroundColor: gradientStroke,
    //     borderColor: '#00d6b4',
    //     borderWidth: 2,
    //     borderDash: [],
    //     borderDashOffset: 0.0,
    //     pointBackgroundColor: '#00d6b4',
    //     pointBorderColor: 'rgba(255,255,255,0)',
    //     pointHoverBackgroundColor: '#00d6b4',
    //     pointBorderWidth: 20,
    //     pointHoverRadius: 4,
    //     pointHoverBorderWidth: 15,
    //     pointRadius: 4,
    //     data: [90, 27, 60, 12, 80],
    //   }]
    // };

    // var myChart = new Chart(ctxGreen, {
    //   type: 'line',
    //   data: data,
    //   options: gradientChartOptionsConfigurationWithTooltipGreen

    // });



    // var chart_labels = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
    // var chart_data = [100, 70, 90, 70, 85, 60, 75, 60, 90, 80, 110, 100];

    
    var ctx = document.getElementById("chartBig1").getContext('2d');

    var gradientStroke = ctx.createLinearGradient(0, 230, 0, 50);

    const getDatesBetweenDates = (startDate, endDate) => {
      let dates = []
      //to avoid modifying the original date
      const theDate = new Date(startDate)
      while (theDate < endDate) {
        add=new Date(theDate)
        month=String(add.getMonth()+1)
        day=String(add.getDate())
        if (month.length < 2) month = '0' + month;
        if (day.length < 2) day = '0' + day;
        dates = [...dates, day +'/'+month]
        theDate.setDate(theDate.getDate() + 1)
      }
      return dates
    }
    const today = new Date()
    // console.log(today)
    today.setDate(today.getDate()-46)
    console.log(today)
    const sixtydayrange = new Date(today)
    sixtydayrange.setDate( sixtydayrange.getDate() + 60)

    dates=getDatesBetweenDates(today, sixtydayrange)
    const initializeArrayWithRange = (end, start = 0, step = 1) =>
      Array.from(
        { length: Math.ceil((end - start + 1) / step) },
        (_, i) => i * step + start
      );
      chart_labels=initializeArrayWithRange(83)
      var eth_pred_dataset = JSON.parse(document.getElementById("ethDataPred").dataset.gecode);
      var eth_act_dataset = JSON.parse(document.getElementById("ethDataAct").dataset.gecode);
      
      var btc_pred_dataset = JSON.parse(document.getElementById("btcDataPred").dataset.gecode);
      var btc_act_dataset = JSON.parse(document.getElementById("btcDataAct").dataset.gecode);
      
      var xmr_pred_dataset = JSON.parse(document.getElementById("xmrDataPred").dataset.gecode);
      var xmr_act_dataset = JSON.parse(document.getElementById("xmrDataAct").dataset.gecode);
      
      var dataEth = {
        labels:chart_labels,
        datasets: [{
          label: "Prediction",
          fill: true,
          backgroundColor: gradientStroke,
          borderColor: '#d048b6',
          borderWidth: 2.2,
          borderDash: [],
          borderDashOffset: 0.0,
          pointBackgroundColor: '#d048b6',
          pointBorderColor: 'rgba(255,66,255,0.11)',
          pointHoverBackgroundColor: '#d048b6',
          pointBorderWidth: 18,
          pointHoverRadius: 3,
          pointHoverBorderWidth: 15,
          pointRadius: 2.3,
          data:eth_pred_dataset  
          
          }
        ,{
          label: "Actual",
          fill: true,
          backgroundColor: gradientStroke,
          borderColor: '#45b1e8 ',
          borderWidth: 2.2,
          borderDash: [],
          borderDashOffset: 0.0,
          pointBackgroundColor: '#45b1e8 ',
          pointBorderColor: 'rgba(72,72,176,0.24)',
          pointHoverBackgroundColor: '#45b1e8 ',
          pointBorderWidth: 18,
          pointHoverRadius: 3,
          pointHoverBorderWidth: 15,
          pointRadius: 2.3,
          data: eth_act_dataset
          
         }
      ]
        
      };
 

    gradientStroke.addColorStop(1, 'rgba(72,72,176,0.1)');
    gradientStroke.addColorStop(0.4, 'rgba(72,72,176,0.0)');
    gradientStroke.addColorStop(0, 'rgba(119,52,169,0)'); //purple colors
    var config = {
      type: 'line',
      data: dataEth,
      options: gradientChartOptionsConfigurationWithTooltipPurple,
    };
    
 
    var predictions = new Chart(ctx, config);
    $("#0").click(function() {
      var data = predictions.config.data;
      data.datasets[0].data = eth_pred_dataset;
      data.datasets[1].data = eth_act_dataset;
      data.labels = chart_labels;
      predictions.update();
      document.getElementById('cryptoNameForcase').innerHTML="Ethereum"

    });
    $("#1").click(function() {
      var data = predictions.config.data;
      data.datasets[0].data = btc_pred_dataset;
      data.datasets[1].data = btc_act_dataset;
      data.labels = chart_labels;
      predictions.update();
      document.getElementById('cryptoNameForcase').innerHTML="Bitcoin"


    });

    $("#2").click(function() {
      var data = predictions.config.data;
      data.datasets[0].data = xmr_pred_dataset;
      data.datasets[1].data = xmr_act_dataset;
      data.labels = chart_labels;
      predictions.update();
      document.getElementById('cryptoNameForcase').innerHTML="Monero"

    });

  const symbols =["BTC","ETH","XMR"]
  fetch("https://min-api.cryptocompare.com/data/pricemultifull?fsyms=BTC,ETH,XMR,LUNA,XRP,SOL,BNB,TRX&tsyms=USD&api_key={ea0232c4ea8a3007655f1518de6af8ea6c4a5e546ddf83988ec885db9600a11e}")
    .then(response => response.json())
    .then(data => data.DISPLAY)
    .then(data => {
      let table = document.getElementById("DailyChart")
      let row = table.insertRow(0);
      let symbol = row.insertCell(0)
      symbol.innerHTML = data.BTC.USD.FROMSYMBOL
      let price = row.insertCell(1)
      price.innerHTML = data.BTC.USD.PRICE
      let change = row.insertCell(2)
      change.innerHTML = data.BTC.USD.CHANGEDAY
      let perct_change = row.insertCell(3)
      perct_change.innerHTML = data.BTC.USD.CHANGEPCTDAY
      let open = row.insertCell(4)
      open.innerHTML = data.BTC.USD.OPENDAY
      let high = row.insertCell(5)
      high.innerHTML = data.BTC.USD.HIGHDAY
      let low = row.insertCell(6)
      low.innerHTML = data.BTC.USD.LOWDAY
      let volume = row.insertCell(7)
      volume.innerHTML = data.BTC.USD.VOLUMEDAY

      row = table.insertRow(1)
      symbol = row.insertCell(0)
      symbol.innerHTML = data.ETH.USD.FROMSYMBOL
      price = row.insertCell(1)
      price.innerHTML = data.ETH.USD.PRICE
      change = row.insertCell(2)
      change.innerHTML = data.ETH.USD.CHANGEDAY
      perct_change = row.insertCell(3)
      perct_change.innerHTML = data.ETH.USD.CHANGEPCTDAY
      open = row.insertCell(4)
      open.innerHTML = data.ETH.USD.OPENDAY
      high = row.insertCell(5)
      high.innerHTML = data.ETH.USD.HIGHDAY
      low = row.insertCell(6)
      low.innerHTML = data.ETH.USD.LOWDAY
      volume = row.insertCell(7)
      volume.innerHTML = data.ETH.USD.VOLUMEDAY

      row = table.insertRow(2)
      symbol = row.insertCell(0)
      symbol.innerHTML = data.XMR.USD.FROMSYMBOL
      price = row.insertCell(1)
      price.innerHTML = data.XMR.USD.PRICE
      change = row.insertCell(2)
      change.innerHTML = data.XMR.USD.CHANGEDAY
      perct_change = row.insertCell(3)
      perct_change.innerHTML = data.XMR.USD.CHANGEPCTDAY
      open = row.insertCell(4)
      open.innerHTML = data.XMR.USD.OPENDAY
      high = row.insertCell(5)
      high.innerHTML = data.XMR.USD.HIGHDAY
      low = row.insertCell(6)
      low.innerHTML = data.XMR.USD.LOWDAY
      volume = row.insertCell(7)
      volume.innerHTML = data.XMR.USD.VOLUMEDAY

      row = table.insertRow(3)
      symbol = row.insertCell(0)
      symbol.innerHTML = data.LUNA.USD.FROMSYMBOL
      price = row.insertCell(1)
      price.innerHTML = data.LUNA.USD.PRICE
      change = row.insertCell(2)
      change.innerHTML = data.LUNA.USD.CHANGEDAY
      perct_change = row.insertCell(3)
      perct_change.innerHTML = data.LUNA.USD.CHANGEPCTDAY
      open = row.insertCell(4)
      open.innerHTML = data.LUNA.USD.OPENDAY
      high = row.insertCell(5)
      high.innerHTML = data.LUNA.USD.HIGHDAY
      low = row.insertCell(6)
      low.innerHTML = data.LUNA.USD.LOWDAY
      volume = row.insertCell(7)
      volume.innerHTML = data.LUNA.USD.VOLUMEDAY

      row = table.insertRow(4)
      symbol = row.insertCell(0)
      symbol.innerHTML = data.XRP.USD.FROMSYMBOL
      price = row.insertCell(1)
      price.innerHTML = data.XRP.USD.PRICE
      change = row.insertCell(2)
      change.innerHTML = data.XRP.USD.CHANGEDAY
      perct_change = row.insertCell(3)
      perct_change.innerHTML = data.XRP.USD.CHANGEPCTDAY
      open = row.insertCell(4)
      open.innerHTML = data.XRP.USD.OPENDAY
      high = row.insertCell(5)
      high.innerHTML = data.XRP.USD.HIGHDAY
      low = row.insertCell(6)
      low.innerHTML = data.XRP.USD.LOWDAY
      volume = row.insertCell(7)
      volume.innerHTML = data.XRP.USD.VOLUMEDAY

      row = table.insertRow(5)
      symbol = row.insertCell(0)
      symbol.innerHTML = data.SOL.USD.FROMSYMBOL
      price = row.insertCell(1)
      price.innerHTML = data.SOL.USD.PRICE
      change = row.insertCell(2)
      change.innerHTML = data.SOL.USD.CHANGEDAY
      perct_change = row.insertCell(3)
      perct_change.innerHTML = data.SOL.USD.CHANGEPCTDAY
      open = row.insertCell(4)
      open.innerHTML = data.SOL.USD.OPENDAY
      high = row.insertCell(5)
      high.innerHTML = data.SOL.USD.HIGHDAY
      low = row.insertCell(6)
      low.innerHTML = data.SOL.USD.LOWDAY
      volume = row.insertCell(7)
      volume.innerHTML = data.SOL.USD.VOLUMEDAY

      row = table.insertRow(6)
      symbol = row.insertCell(0)
      symbol.innerHTML = data.BNB.USD.FROMSYMBOL
      price = row.insertCell(1)
      price.innerHTML = data.BNB.USD.PRICE
      change = row.insertCell(2)
      change.innerHTML = data.BNB.USD.CHANGEDAY
      perct_change = row.insertCell(3)
      perct_change.innerHTML = data.BNB.USD.CHANGEPCTDAY
      open = row.insertCell(4)
      open.innerHTML = data.BNB.USD.OPENDAY
      high = row.insertCell(5)
      high.innerHTML = data.BNB.USD.HIGHDAY
      low = row.insertCell(6)
      low.innerHTML = data.BNB.USD.LOWDAY
      volume = row.insertCell(7)
      volume.innerHTML = data.BNB.USD.VOLUMEDAY

      row = table.insertRow(7)
      symbol = row.insertCell(0)
      symbol.innerHTML = data.TRX.USD.FROMSYMBOL
      price = row.insertCell(1)
      price.innerHTML = data.TRX.USD.PRICE
      change = row.insertCell(2)
      change.innerHTML = data.TRX.USD.CHANGEDAY
      perct_change = row.insertCell(3)
      perct_change.innerHTML = data.TRX.USD.CHANGEPCTDAY
      open = row.insertCell(4)
      open.innerHTML = data.TRX.USD.OPENDAY
      high = row.insertCell(5)
      high.innerHTML = data.TRX.USD.HIGHDAY
      low = row.insertCell(6)
      low.innerHTML = data.TRX.USD.LOWDAY
      volume = row.insertCell(7)
      volume.innerHTML = data.TRX.USD.VOLUMEDAY

      
    })



       
       
      

  },

  
  showNotification: function(from, align) {
    color = Math.floor((Math.random() * 4) + 1);

    $.notify({
      icon: "tim-icons icon-bell-55",
      message: "Welcome to <b>Black Dashboard</b> - a beautiful freebie for every web developer."

    }, {
      type: type[color],
      timer: 8000,
      placement: {
        from: from,
        align: align
      }
    });
  }

};

