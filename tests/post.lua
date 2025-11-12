-- post.lua
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

-- Example input for your Iris model
local body = '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

wrk.body = body

-- Optional: include a random variation to avoid caching (optional but good for realism)
request = function()
    return wrk.format(nil, nil, nil, wrk.body)
end
