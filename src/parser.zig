const std = @import("std");

pub const Location = struct {
    row: i32,
    col: i32,
};

const TokenType = enum {
    INTEGER,
    FLOAT,
    IDENT,
    OPERATION,
    LPAREN,
    RPAREN,
    EQUAL,
    COLON,
    SEMI,
    BIT_LSH,
    BIT_RSH,
    BIT_AND,
    BIT_XOR,
    BIT_OR,
    NO_OP,
    EOF,
};

pub const Token = struct {
    type: TokenType,
    value: []const u8,
    loc: Location,

    pub fn EOF() Token {
        return .{
            .type = .EOF,
            .value = "",
            .loc = .{ .row = 0, .col = 0 },
        };
    }
};

const Lexer = struct {
    content: []u8,
    index: usize,
    char: u8,
    row: i32,
    col: i32,

    pub fn new(content: []u8) Lexer {
        return Lexer{
            .content = content,
            .index = 0,
            .char = content[0],
            .row = 1,
            .col = 1,
        };
    }

    pub fn getNextToken(self: *Lexer) Token {
        self.skipWhitespace();

        if (self.index == self.content.len - 1) {
            return Token{
                .type = .EOF,
                .value = "",
                .loc = .{ .row = self.row, .col = self.col },
            };
        }

        if (std.ascii.isDigit(self.char)) {
            return self.readNumeric();
        }

        if (std.ascii.isAlphabetic(self.char) or self.char == '_') {
            return self.readIdent();
        }

        if (self.char == '<' and self.peek() == '<') {
            self.advance();
            return self.advanceWithToken(.BIT_LSH, "<<");
        }

        if (self.char == '>' and self.peek() == '>') {
            self.advance();
            return self.advanceWithToken(.BIT_RSH, ">>");
        }

        return switch (self.char) {
            '+', '-', '*', '/' => self.advanceWithToken(TokenType.OPERATION, toSlice(self.char, std.heap.page_allocator)),
            '(' => self.advanceWithToken(.LPAREN, toSlice(self.char, std.heap.page_allocator)),
            ')' => self.advanceWithToken(.RPAREN, toSlice(self.char, std.heap.page_allocator)),
            '&' => self.advanceWithToken(.BIT_AND, toSlice(self.char, std.heap.page_allocator)),
            '^' => self.advanceWithToken(.BIT_XOR, toSlice(self.char, std.heap.page_allocator)),
            '|' => self.advanceWithToken(.BIT_OR, toSlice(self.char, std.heap.page_allocator)),
            ';' => self.advanceWithToken(.SEMI, toSlice(self.char, std.heap.page_allocator)),
            ':' => self.advanceWithToken(.COLON, toSlice(self.char, std.heap.page_allocator)),
            '=' => self.advanceWithToken(.EQUAL, toSlice(self.char, std.heap.page_allocator)),
            0 => self.advanceWithToken(.EOF, ""),
            else => self.advanceWithToken(TokenType.NO_OP, toSlice(self.char, std.heap.page_allocator)),
        };
    }

    fn readIdent(self: *Lexer) Token {
        var chars = std.ArrayList(u8).init(std.heap.page_allocator);

        while (std.ascii.isAlphabetic(self.char) or std.ascii.isDigit(self.char) or self.char == '_') {
            chars.append(self.char) catch {
                return Token.EOF();
            };
            self.advance();
        }

        const value = chars.toOwnedSlice() catch {
            return Token.EOF();
        };

        return Token{ .type = .IDENT, .value = value, .loc = .{ .row = self.row, .col = self.col } };
    }

    fn readNumeric(self: *Lexer) Token {
        var chars = std.ArrayList(u8).init(std.heap.page_allocator);

        while (std.ascii.isDigit(self.char) or self.char == '.') {
            chars.append(self.char) catch {
                return Token.EOF();
            };
            self.advance();
        }

        var isFloat: bool = false;
        for (chars.items) |char| {
            if (char == '.') {
                isFloat = true;
            }
        }

        const tokenType: TokenType = if (isFloat) .FLOAT else .INTEGER;

        const value = chars.toOwnedSlice() catch {
            return Token.EOF();
        };

        return Token{ .type = tokenType, .value = value, .loc = .{ .col = self.col, .row = self.row } };
    }

    fn skipWhitespace(self: *Lexer) void {
        while (self.index < self.content.len - 1 and std.ascii.isWhitespace(self.char)) {
            self.advance();
        }
    }

    fn peek(self: *Lexer) u8 {
        if (self.index < self.content.len - 1) {
            return self.content[self.index + 1];
        }
        return 0;
    }

    fn advanceWithToken(self: *Lexer, tokenType: TokenType, value: []const u8) Token {
        const token = Token{ .type = tokenType, .value = value, .loc = .{ .col = self.col, .row = self.row } };
        self.advance();
        return token;
    }

    fn advance(self: *Lexer) void {
        if (self.index < self.content.len - 1) {
            if (self.char == '\n') {
                self.row += 1;
                self.col = 0;
            }

            self.col += 1;
            self.index += 1;
            self.char = self.content[self.index];
        } else {
            self.char = 0;
        }
    }
};

const AstNodeTag = enum {
    binOp,
    number,
    ident,
    varDecl,
    varAssign,
    program,
};

pub const Ast = struct {
    pub const Node = union(AstNodeTag) {
        pub const BinOp = struct {
            lhs: Ast.Node,
            op: Token,
            rhs: Ast.Node,
            loc: Location,
        };
        pub const Number = struct {
            value: Token,
            type: Type,
            loc: Location,
            pub const Type = enum {
                float,
                int,
            };
        };
        pub const Ident = struct {
            value: Token,
            loc: Location,
        };
        pub const VarDecl = struct {
            ident: Ast.Node.Ident,
            value: Ast.Node,
            loc: Location,
        };
        pub const VarAssign = struct {
            ident: Ast.Node.Ident,
            value: Ast.Node,
            loc: Location,
        };
        pub const Program = struct {
            expressions: std.ArrayList(Ast.Node),
        };
        binOp: *Ast.Node.BinOp,
        number: *Ast.Node.Number,
        ident: *Ast.Node.Ident,
        varDecl: *Ast.Node.VarDecl,
        varAssign: *Ast.Node.VarAssign,
        program: *Ast.Node.Program,

        pub fn getTag(self: *const Ast.Node) AstNodeTag {
            return @as(AstNodeTag, self.*);
        }

        pub fn allocBinOp(lhs: *Ast.Node, op: Token, rhs: *Ast.Node, loc: Location, alloc: std.mem.Allocator) *Ast.Node.BinOp {
            var mem = alloc.create(Ast.Node.BinOp) catch {
                std.debug.panic("Could not create new BinOp ast node\n", .{});
            };
            mem.lhs = lhs.*;
            mem.op = op;
            mem.rhs = rhs.*;
            mem.loc = loc;

            return mem;
        }

        pub fn allocNumber(value: Token, numType: Ast.Node.Number.Type, loc: Location, alloc: std.mem.Allocator) *Ast.Node.Number {
            var mem = alloc.create(Ast.Node.Number) catch {
                std.debug.panic("Could not create new Number ast node\n", .{});
            };
            mem.value = value;
            mem.type = numType;
            mem.loc = loc;

            return mem;
        }

        pub fn allocIdent(value: Token, loc: Location, alloc: std.mem.Allocator) *Ast.Node.Ident {
            var mem = alloc.create(Ast.Node.Ident) catch {
                std.debug.panic("Could not create new Number ast node\n", .{});
            };
            mem.value = value;
            mem.loc = loc;

            return mem;
        }

        pub fn allocVarDecl(ident: *Ident, value: *const Ast.Node, loc: Location, alloc: std.mem.Allocator) *Ast.Node.VarDecl {
            var mem = alloc.create(Ast.Node.VarDecl) catch {
                std.debug.panic("Could not create new VarDecl ast node\n", .{});
            };
            mem.ident = ident.*;
            mem.value = value.*;
            mem.loc = loc;

            return mem;
        }

        pub fn allocVarAssign(ident: *Ident, value: *const Ast.Node, loc: Location, alloc: std.mem.Allocator) *Ast.Node.VarAssign {
            var mem = alloc.create(Ast.Node.VarAssign) catch {
                std.debug.panic("Could not create new VarAssign ast node\n", .{});
            };
            mem.ident = ident.*;
            mem.value = value.*;
            mem.loc = loc;

            return mem;
        }
    };
    head: Node.Program,
};

pub const Parser = struct {
    lexer: *Lexer,
    k: usize,
    tokens: []Token,
    index: usize,
    allocator: std.mem.Allocator,

    pub fn new(input: []u8, maxTokens: comptime_int, allocator: std.mem.Allocator) !Parser {
        var heapLexer = try allocator.alloc(Lexer, 1);
        heapLexer[0] = Lexer.new(input);
        var parser = Parser{ .lexer = &heapLexer[0], .k = maxTokens, .tokens = try allocator.alloc(Token, maxTokens), .index = 0, .allocator = allocator };

        for (0..maxTokens) |i| {
            parser.tokens[i] = parser.lexer.getNextToken();
        }

        return parser;
    }

    pub fn parseInput(self: *Parser) Ast {
        const list = std.ArrayList(Ast.Node).init(self.allocator);
        var program = Ast.Node.Program{
            .expressions = list,
        };
        while (self.current().type != .EOF) {
            const expr = switch (self.current().type) {
                .IDENT => switch (self.peek().type) {
                    .COLON => self.parseVarDecl(),
                    .EQUAL => self.parseVarAssign(),
                    else => {
                        std.debug.print("{d}:{d}: ERROR: Invalid token after identifer. Found: {any}\n", .{ self.peek().loc.row, self.peek().loc.col, self.peek().type });
                        std.process.exit(1);
                    },
                },
                TokenType.LPAREN, TokenType.FLOAT, TokenType.INTEGER => self.parseExpr(),
                else => {
                    std.debug.print("{d}:{d}: ERROR: Invalid token. Found: {any}\n", .{ self.current().loc.row, self.current().loc.col, self.current().type });
                    std.process.exit(1);
                },
            };

            self.consume(&[_]TokenType{.SEMI});

            program.expressions.append(expr) catch {
                std.debug.print("Could not append new Ast node to program list. Exiting...\n", .{});
                std.process.exit(1);
            };
            //std.debug.print("Expression #{d}: {any}\n", .{ program.expressions.items.len, expr });
            //std.debug.print("Next token: {any}\n", .{self.peek()});
        }
        return Ast{ .head = program };
    }

    fn parseVarDecl(self: *Parser) Ast.Node {
        // std.debug.print("parse var decl\n", .{});
        std.debug.assert(self.current().type == .IDENT);
        const varNode = self.parseExpr();

        self.consume(&[_]TokenType{.COLON});
        self.consume(&[_]TokenType{.EQUAL});

        const value = self.parseExpr();

        return Ast.Node{ .varDecl = Ast.Node.allocVarDecl(varNode.ident, &value, varNode.ident.loc, self.allocator) };
    }

    fn parseVarAssign(self: *Parser) Ast.Node {
        std.debug.assert(self.current().type == .IDENT);
        const varNode = self.parseExpr();

        self.consume(&[_]TokenType{.EQUAL});

        const value = self.parseExpr();

        return Ast.Node{ .varAssign = Ast.Node.allocVarAssign(varNode.ident, &value, varNode.ident.loc, self.allocator) };
    }

    fn parseExpr(self: *Parser) Ast.Node {
        // std.debug.print("parse expr\n", .{});
        var iterate = true;
        const startLoc = self.current().loc;
        var lhs = self.parseShift();

        while (iterate) {
            switch (self.current().type) {
                .BIT_AND, .BIT_XOR, .BIT_OR => {
                    const op = self.current();
                    self.consume(&[_]TokenType{ .BIT_AND, .BIT_XOR, .BIT_OR });
                    if (self.expect(&[_]TokenType{ .LPAREN, .FLOAT, .INTEGER, .IDENT })) {
                        var rhs = self.parseShift();
                        lhs = Ast.Node{
                            .binOp = Ast.Node.allocBinOp(&lhs, op, &rhs, startLoc, self.allocator),
                        };
                    }
                },
                else => iterate = false,
            }
        }

        return lhs;
    }

    fn parseShift(self: *Parser) Ast.Node {
        // std.debug.print("parse shift\n", .{});
        var iterate = true;
        const startLoc = self.current().loc;
        var lhs = self.parseAdditive();

        while (iterate) {
            switch (self.current().type) {
                .BIT_LSH, .BIT_RSH => {
                    const op = self.current();
                    self.consume(&[_]TokenType{ .BIT_LSH, .BIT_RSH });
                    if (self.expect(&[_]TokenType{ .LPAREN, .FLOAT, .INTEGER, .IDENT })) {
                        var rhs = self.parseAdditive();
                        lhs = Ast.Node{
                            .binOp = Ast.Node.allocBinOp(&lhs, op, &rhs, startLoc, self.allocator),
                        };
                    }
                },
                else => iterate = false,
            }
        }

        return lhs;
    }

    fn parseAdditive(self: *Parser) Ast.Node {
        // std.debug.print("parse additive\n", .{});
        var iterate = true;
        const startLoc = self.current().loc;
        var lhs = self.parseTerm();

        while (iterate) {
            switch (self.current().type) {
                .OPERATION => {
                    if (std.mem.eql(u8, self.current().value, "+") or std.mem.eql(u8, self.current().value, "-")) {
                        const op = self.current();
                        self.consume(&[_]TokenType{.OPERATION});
                        if (self.expect(&[_]TokenType{ .LPAREN, .FLOAT, .INTEGER, .IDENT })) {
                            var rhs = self.parseTerm();
                            lhs = Ast.Node{
                                .binOp = Ast.Node.allocBinOp(&lhs, op, &rhs, startLoc, self.allocator),
                            };
                        }
                    } else iterate = false;
                },
                else => iterate = false,
            }
        }

        return lhs;
    }

    fn parseTerm(self: *Parser) Ast.Node {
        // std.debug.print("parse term\n", .{});
        var iterate = true;
        const startLoc = self.current().loc;
        var lhs = self.parseFactor();

        while (iterate) {
            switch (self.current().type) {
                .OPERATION => {
                    if (std.mem.eql(u8, self.current().value, "*") or std.mem.eql(u8, self.current().value, "/")) {
                        const op = self.current();
                        self.consume(&[_]TokenType{.OPERATION});
                        if (self.expect(&[_]TokenType{ .LPAREN, .FLOAT, .INTEGER, .IDENT })) {
                            var rhs = self.parseFactor();
                            lhs = Ast.Node{
                                .binOp = Ast.Node.allocBinOp(&lhs, op, &rhs, startLoc, self.allocator),
                            };
                        }
                    } else iterate = false;
                },
                else => iterate = false,
            }
        }

        return lhs;
    }

    fn parseFactor(self: *Parser) Ast.Node {
        // std.debug.print("parse factor\n", .{});
        const token = self.current();
        var res: Ast.Node = undefined;

        if (token.type == .LPAREN) {
            self.consume(&[_]TokenType{.LPAREN});
            res = self.parseExpr();
        } else if (token.type == .FLOAT) {
            res = Ast.Node{ .number = Ast.Node.allocNumber(token, .float, token.loc, self.allocator) };
        } else if (token.type == .INTEGER) {
            res = Ast.Node{ .number = Ast.Node.allocNumber(token, .int, token.loc, self.allocator) };
        } else if (token.type == .IDENT) {
            res = Ast.Node{ .ident = Ast.Node.allocIdent(token, token.loc, self.allocator) };
        } else if (token.type == .NO_OP) {
            self.consume(&[_]TokenType{.NO_OP});
            return self.parseFactor();
        }
        self.consume(&[_]TokenType{ .RPAREN, .FLOAT, .INTEGER, .IDENT });
        // std.debug.print("Factor: {any}\n", .{res});
        return res;
    }

    fn current(self: *Parser) Token {
        return self.tokens[self.index % self.k];
    }

    fn peek(self: *Parser) Token {
        return self.tokens[(self.index + 1) % self.k];
    }

    fn consume(self: *Parser, expected: []const TokenType) void {
        _ = self.expect(expected);

        self.tokens[self.index % self.k] = self.lexer.getNextToken();
        self.index += 1;

        // std.debug.print("next token: {any} '{s}'\n", .{ self.current().type, self.current().value });
    }

    fn expect(self: *Parser, expected: []const TokenType) bool {
        var validToken: bool = false;
        for (0..expected.len) |i| {
            if (self.current().type == expected[i]) {
                validToken = true;
            }
        }
        if (!validToken) {
            std.debug.print("{d}:{d}: ERROR: Invalid token. Found: {any}, expected: {any}\n", .{ self.current().loc.row, self.current().loc.col, self.current().type, expected });
            std.process.exit(1);
        }
        return validToken;
    }
};

fn toSlice(char: u8, allocator: std.mem.Allocator) []const u8 {
    const slice = allocator.alloc(u8, 1) catch unreachable;
    slice[0] = char;
    return slice;
}
