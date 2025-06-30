library(shiny)
library(ggplot2)
library(dplyr)
library(plotly)
library(grid)
library('RCurl')

# Load the dataset
data <- read.csv("FundingMapData.csv")

# Get numeric columns for selection
numeric_columns <- c("SumWeights", "UMAP1", "UMAP2", "Size", "StarCount", "Forks", "IssueCount")

# Define UI for the application
ui <- fluidPage(
  titlePanel("Deep Funding Cartography with the Omniacs.DAO"),
  
  sidebarLayout(
    sidebarPanel(
      # div(
      #   tags$img(src = "server_Icon1.png", width = "30%", alt = "Logo"),
      #   style = "text-align: center; margin-bottom: 20px;"
      # ),
      
      selectInput("xAxis", 
                  "Select X-axis:", 
                  choices = numeric_columns, 
                  selected = "UMAP1"),
      
      selectInput("yAxis", 
                  "Select Y-axis:", 
                  choices = numeric_columns, 
                  selected = "UMAP2"),
      
      selectInput("filterColumn", 
                  "Select Column to Highlight:", 
                  choices = numeric_columns, 
                  selected = "SumWeights"),
      
      sliderInput("percentileRange", 
                  "Select Percentile Range:", 
                  min = 0, 
                  max = 100, 
                  value = c(10, 90),
                  step = 5)
    ),
    
    mainPanel(
      plotlyOutput("umapPlot", height = "700px")  # Increased plot size
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  highlightedData <- reactive({
    req(input$filterColumn, input$percentileRange)
    
    # Calculate the percentile values based on the selected column
    lower_cutoff <- quantile(data[[input$filterColumn]], input$percentileRange[1] / 100, na.rm = TRUE)
    upper_cutoff <- quantile(data[[input$filterColumn]], input$percentileRange[2] / 100, na.rm = TRUE)
    
    data %>%
      mutate(HighlightCategory = case_when(
        get(input$filterColumn) > upper_cutoff ~ "Above Upper Percentile",
        get(input$filterColumn) < lower_cutoff ~ "Below Lower Percentile",
        TRUE ~ "Middle Range"
      ))
  })
  
  output$umapPlot <- renderPlotly({
    plot_data <- highlightedData()
    
    p <- ggplot(plot_data, aes_string(x = input$xAxis, y = input$yAxis, color = "HighlightCategory", text = "Value", text2 = input$filterColumn)) +
      geom_point(alpha = 0.6, size = 2) +
      scale_color_manual(values = c("Above Upper Percentile" = "green", 
                                    "Below Lower Percentile" = "red", 
                                    "Middle Range" = "black")) +
      labs(title = "Dynamic UMAP Plot with Percentile Highlighting",
           x = input$xAxis,
           y = input$yAxis,
           color = "Highlight Category") +
      theme_minimal() 
    
    plotlyfig <- ggplotly(p, tooltip = c("text", "text2"))
    image_file <- "www/server_Icon1.png"
    txt <- RCurl::base64Encode(readBin(image_file, "raw", file.info(image_file)[1, "size"]), "txt")
    plotlyfig %>%
    layout(
    images =
      list(source = paste('data:image/png;base64', txt, sep=','),
           xref = "paper",
           yref = "paper",
           x= 1,
           y= 0.1,
           sizex = 0.1,
           sizey = 0.1,
           opacity = 0.8
      ))
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
